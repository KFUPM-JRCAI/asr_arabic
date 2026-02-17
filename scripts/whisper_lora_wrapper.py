#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for LoRA/PEFT-adapted Whisper models (transformers-based)."""

from __future__ import annotations

import argparse
import cgi
import json
import os
import tempfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple, cast

import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel


@dataclass
class WhisperLoraConfig:
    base_model_id: str
    adapter_id: str
    device: str
    torch_dtype: torch.dtype
    language: str


# Global model and processor (loaded once at startup)
_MODEL = None
_PROCESSOR = None


def load_model(cfg: WhisperLoraConfig) -> Tuple[PeftModel, AutoProcessor]:
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    print(f"Loading base model {cfg.base_model_id}...")
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        cfg.base_model_id,
        torch_dtype=cfg.torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    print(f"Loading LoRA adapter {cfg.adapter_id}...")
    model = PeftModel.from_pretrained(base_model, cfg.adapter_id)
    model = model.merge_and_unload()
    model.to(cfg.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(cfg.base_model_id)

    _MODEL = model
    _PROCESSOR = processor
    print(f"Model loaded: {cfg.base_model_id} + {cfg.adapter_id}")
    return _MODEL, _PROCESSOR


def transcribe_audio(
    audio_path: str,
    model,
    processor: AutoProcessor,
    device: str,
    torch_dtype: torch.dtype,
    language: str = "ar",
) -> str:
    """Transcribe audio file using direct model inference."""
    audio_array, _sr = librosa.load(audio_path, sr=16000)

    input_features = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device, dtype=torch_dtype)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=440,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


class WhisperLoraServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        cfg: WhisperLoraConfig,
        auth_token: Optional[str],
        display_name: str,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.cfg = cfg
        self.auth_token = auth_token
        self.model_name = display_name
        self.model, self.processor = load_model(cfg)


class OpenAICompatHandler(BaseHTTPRequestHandler):
    def _srv(self) -> WhisperLoraServer:
        return cast(WhisperLoraServer, self.server)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

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

    # ---- GET ----------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in ("/v1/models", "/v1/models/"):
            self._json_response(200, {
                "object": "list",
                "data": [{
                    "id": self._srv().model_name,
                    "object": "model",
                    "owned_by": "community",
                }],
            })
            return
        if self.path.startswith("/v1/models/"):
            model_id = self.path.replace("/v1/models/", "").rstrip("/")
            if model_id == self._srv().model_name or \
               model_id.replace("/", "-") == self._srv().model_name.replace("/", "-"):
                self._json_response(200, {
                    "id": self._srv().model_name,
                    "object": "model",
                    "owned_by": "community",
                })
                return
        if self.path in ("/health", "/"):
            self._json_response(200, {"status": "ok"})
            return
        self._json_response(404, {"error": "not found"})

    # ---- POST ---------------------------------------------------------
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

            import shutil
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file_item.file, tmp)
                tmp_path = Path(tmp.name)

            try:
                language = form.getvalue("language", self._srv().cfg.language)
                srv = self._srv()
                transcription = transcribe_audio(
                    audio_path=str(tmp_path),
                    model=srv.model,
                    processor=srv.processor,
                    device=srv.cfg.device,
                    torch_dtype=srv.cfg.torch_dtype,
                    language=language,
                )
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


# ---- CLI ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listen-host", default=os.getenv("LISTEN_HOST", "0.0.0.0"))
    p.add_argument("--listen-port", type=int, default=int(os.getenv("LISTEN_PORT", "8099")))
    p.add_argument(
        "--base-model",
        default=os.getenv("BASE_MODEL_ID", "openai/whisper-large-v3"),
        help="HuggingFace ID of the base Whisper model",
    )
    p.add_argument(
        "--adapter",
        default=os.getenv("ADAPTER_ID", "RaedMughaus/whisper-large-v3-finetuned-cv-corpus-24-ar"),
        help="HuggingFace ID of the PEFT/LoRA adapter",
    )
    p.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME"),
        help="Display name for the model (defaults to --adapter value)",
    )
    p.add_argument("--device", default=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--language", default=os.getenv("LANGUAGE", "ar"))
    p.add_argument("--auth-token", default=os.getenv("WRAPPER_AUTH_TOKEN"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    torch_dtype = torch.float16 if "cuda" in device else torch.float32
    display_name = args.model_name or args.adapter

    cfg = WhisperLoraConfig(
        base_model_id=args.base_model,
        adapter_id=args.adapter,
        device=device,
        torch_dtype=torch_dtype,
        language=args.language,
    )

    print(f"Starting whisper-lora wrapper on {args.listen_host}:{args.listen_port}")
    print(f"Base: {cfg.base_model_id}  Adapter: {cfg.adapter_id}")

    server = WhisperLoraServer(
        (args.listen_host, args.listen_port),
        OpenAICompatHandler,
        cfg,
        args.auth_token,
        display_name,
    )

    print("Server ready!")
    server.serve_forever()


if __name__ == "__main__":
    main()
