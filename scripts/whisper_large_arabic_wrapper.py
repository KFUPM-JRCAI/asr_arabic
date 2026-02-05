#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for KFUPM-JRCAI/WhisperLargeArabic (transformers-based Whisper)."""

from __future__ import annotations

import argparse
import cgi
import json
import os
import tempfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Optional, cast, Tuple

import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


@dataclass
class WhisperConfig:
    model_id: str
    device: str
    torch_dtype: torch.dtype
    language: str


# Global model and processor (loaded once at startup)
_MODEL = None
_PROCESSOR = None


def load_model(cfg: WhisperConfig) -> Tuple[AutoModelForSpeechSeq2Seq, AutoProcessor]:
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    print(f"Loading model {cfg.model_id}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(cfg.device)

    processor = AutoProcessor.from_pretrained(cfg.model_id)

    _MODEL = model
    _PROCESSOR = processor
    print(f"Model {cfg.model_id} loaded successfully!")
    return _MODEL, _PROCESSOR


def transcribe_audio(
    audio_path: str,
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    device: str,
    torch_dtype: torch.dtype,
    language: str = "ar",
) -> str:
    """Transcribe audio file using direct model inference (avoids pipeline bugs)."""
    # Load audio with librosa (Whisper expects 16kHz)
    audio_array, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Process input features
    input_features = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device, dtype=torch_dtype)
    
    # Generate transcription
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=440,  # Leave room for decoder input tokens (448 total limit)
        )
    
    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


class WhisperWrapperServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        cfg: WhisperConfig,
        auth_token: Optional[str],
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.cfg = cfg
        self.auth_token = auth_token
        self.model_name = cfg.model_id
        # Pre-load the model at startup
        self.model, self.processor = load_model(cfg)


class OpenAICompatHandler(BaseHTTPRequestHandler):
    def _srv(self) -> WhisperWrapperServer:
        return cast(WhisperWrapperServer, self.server)

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - BaseHTTPRequestHandler API
        return  # quiet logs

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text_response(self, status: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _check_auth(self) -> bool:
        expected = self._srv().auth_token
        if not expected:
            return True
        auth_header = self.headers.get("Authorization", "")
        return auth_header == f"Bearer {expected}"

    def do_GET(self) -> None:
        if self.path == "/v1/models" or self.path == "/v1/models/":
            self._json_response(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self._srv().model_name,
                            "object": "model",
                            "owned_by": "kfupm-jrcai",
                        }
                    ],
                },
            )
            return
        if self.path.startswith("/v1/models/"):
            model_id = self.path.replace("/v1/models/", "").rstrip("/")
            if model_id == self._srv().model_name or model_id.replace("/", "-") == self._srv().model_name.replace("/", "-"):
                self._json_response(
                    200,
                    {
                        "id": self._srv().model_name,
                        "object": "model",
                        "owned_by": "kfupm-jrcai",
                    },
                )
                return
        if self.path == "/health" or self.path == "/":
            self._json_response(200, {"status": "ok"})
            return
        self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._check_auth():
            self._json_response(401, {"error": "unauthorized"})
            return

        if self.path in ("/v1/audio/transcriptions", "/v1/audio/transcriptions/"):
            self._handle_transcription()
            return

        self._json_response(404, {"error": "not found"})

    def _handle_transcription(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        ctype, _ = cgi.parse_header(content_type)
        if ctype != "multipart/form-data":
            self._json_response(400, {"error": "expected multipart/form-data"})
            return

        try:
            # Use cgi.FieldStorage for reliable multipart parsing
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )

            if "file" not in form:
                self._json_response(400, {"error": "no file provided"})
                return

            file_item = form["file"]
            suffix = Path(file_item.filename or "audio.wav").suffix or ".wav"

            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                import shutil
                shutil.copyfileobj(file_item.file, tmp)
                tmp_path = Path(tmp.name)

            try:
                # Get language from form fields
                language = form.getvalue("language", self._srv().cfg.language)

                # Transcribe using direct model inference (avoids pipeline bugs)
                srv = self._srv()
                transcription = transcribe_audio(
                    audio_path=str(tmp_path),
                    model=srv.model,
                    processor=srv.processor,
                    device=srv.cfg.device,
                    torch_dtype=srv.cfg.torch_dtype,
                    language=language,
                )

                # Check response format
                response_format = form.getvalue("response_format", "text")
                if response_format == "json":
                    self._json_response(200, {"text": transcription})
                else:
                    self._text_response(200, transcription)

            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json_response(500, {"error": str(e)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listen-host",
        default=os.getenv("LISTEN_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=int(os.getenv("LISTEN_PORT", "8099")),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_ID", "KFUPM-JRCAI/WhisperLargeArabic"),
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--language",
        default=os.getenv("LANGUAGE", "ar"),
    )
    parser.add_argument(
        "--auth-token",
        default=os.getenv("WRAPPER_AUTH_TOKEN"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    cfg = WhisperConfig(
        model_id=args.model,
        device=device,
        torch_dtype=torch_dtype,
        language=args.language,
    )

    print(f"Starting WhisperLargeArabic wrapper on {args.listen_host}:{args.listen_port}")
    print(f"Model: {cfg.model_id}, Device: {cfg.device}")

    server = WhisperWrapperServer(
        (args.listen_host, args.listen_port),
        OpenAICompatHandler,
        cfg,
        args.auth_token,
    )

    print("Server ready!")
    server.serve_forever()


if __name__ == "__main__":
    main()
