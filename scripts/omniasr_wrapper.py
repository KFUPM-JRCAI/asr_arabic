#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for facebook/omniASR-LLM-1B (fairseq2 / omnilingual-asr)."""

from __future__ import annotations

import argparse
import cgi
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple, cast

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Language mapping: ISO 639-1 → ISO 639-3 + script (required by omniASR)
_LANG_MAP = {
    "ar": "ara_Arab",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "cmn_Hans",
}

# Global pipeline (loaded once at startup)
_PIPELINE: Optional[ASRInferencePipeline] = None


def load_pipeline(model_card: str) -> ASRInferencePipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    print(f"Loading omniASR pipeline (model_card={model_card})...")
    _PIPELINE = ASRInferencePipeline(model_card=model_card)
    print(f"Pipeline {model_card} loaded successfully!")
    return _PIPELINE


def transcribe_audio(
    audio_path: str,
    pipeline: ASRInferencePipeline,
    language: str = "ar",
) -> str:
    """Transcribe audio file using omniASR pipeline."""
    lang_code = _LANG_MAP.get(language, language)
    results = pipeline.transcribe([audio_path], lang=[lang_code], batch_size=1)
    return results[0].strip() if results else ""


class WrapperServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: type,
        pipeline: ASRInferencePipeline,
        model_name: str,
        default_language: str,
        auth_token: Optional[str],
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.pipeline = pipeline
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
                            "owned_by": "facebook",
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
                        "owned_by": "facebook",
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
                    pipeline=srv.pipeline,
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
        default=os.getenv("MODEL_ID", "facebook/omniASR-LLM-1B"),
    )
    parser.add_argument(
        "--model-card",
        default=os.getenv("MODEL_CARD", "omniASR_LLM_1B"),
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

    pipeline = load_pipeline(args.model_card)

    print(f"Starting omniASR wrapper on {args.listen_host}:{args.listen_port}")
    print(f"Model: {args.model}, Model card: {args.model_card}")

    server = WrapperServer(
        (args.listen_host, args.listen_port),
        OpenAICompatHandler,
        pipeline=pipeline,
        model_name=args.model,
        default_language=args.language,
        auth_token=args.auth_token,
    )

    print("Server ready!")
    server.serve_forever()


if __name__ == "__main__":
    main()
