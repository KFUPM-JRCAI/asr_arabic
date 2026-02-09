#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for KFUPM-JRCAI/WhisperTurboArabic (CTranslate2 / faster-whisper)."""

from __future__ import annotations

import argparse
import cgi
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple, cast

from faster_whisper import WhisperModel


# Global model (loaded once at startup)
_MODEL: Optional[WhisperModel] = None


def load_model(model_id: str, device: str, compute_type: str) -> WhisperModel:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    print(f"Loading model {model_id} (device={device}, compute_type={compute_type})...")
    _MODEL = WhisperModel(model_id, device=device, compute_type=compute_type)
    print(f"Model {model_id} loaded successfully!")
    return _MODEL


def transcribe_audio(
    audio_path: str,
    model: WhisperModel,
    language: str = "ar",
) -> str:
    """Transcribe audio file using faster-whisper."""
    segments, _info = model.transcribe(audio_path, language=language, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments)


class WrapperServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: type,
        model: WhisperModel,
        model_name: str,
        default_language: str,
        auth_token: Optional[str],
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.model = model
        self.model_name = model_name
        self.default_language = default_language
        self.auth_token = auth_token


class OpenAICompatHandler(BaseHTTPRequestHandler):
    def _srv(self) -> WrapperServer:
        return cast(WrapperServer, self.server)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
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
        if self.path in ("/v1/models", "/v1/models/"):
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
        if self.path in ("/health", "/"):
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
                language = form.getvalue("language", self._srv().default_language)
                srv = self._srv()
                transcription = transcribe_audio(
                    audio_path=str(tmp_path),
                    model=srv.model,
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
        default=os.getenv("MODEL_ID", "KFUPM-JRCAI/WhisperTurboArabic"),
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", "cuda"),
    )
    parser.add_argument(
        "--compute-type",
        default=os.getenv("COMPUTE_TYPE", "float16"),
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

    model = load_model(args.model, args.device, args.compute_type)

    print(f"Starting WhisperTurboArabic wrapper on {args.listen_host}:{args.listen_port}")
    print(f"Model: {args.model}, Device: {args.device}, Compute: {args.compute_type}")

    server = WrapperServer(
        (args.listen_host, args.listen_port),
        OpenAICompatHandler,
        model=model,
        model_name=args.model,
        default_language=args.language,
        auth_token=args.auth_token,
    )

    print("Server ready!")
    server.serve_forever()


if __name__ == "__main__":
    main()
