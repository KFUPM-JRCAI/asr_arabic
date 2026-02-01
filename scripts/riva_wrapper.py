#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for NVIDIA Riva / Parakeet ASR NIM."""

from __future__ import annotations

import argparse
import cgi
import io
import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import IO, Optional, cast

import httpx
import numpy as np
import soundfile as sf

# Minimal mapping from ISO language codes to the BCP-47 tags Riva expects.
_LANG_MAP = {
    "ar": "ar-AR",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "ru": "ru-RU",
    "hi": "hi-IN",
    "pt": "pt-BR",
    "nl": "nl-NL",
    "sv": "sv-SE",
    "da": "da-DK",
    "cs": "cs-CZ",
    "tr": "tr-TR",
    "pl": "pl-PL",
    "he": "he-IL",
}


@dataclass
class RivaConfig:
    base_url: str
    model: Optional[str]
    default_language: Optional[str]
    timeout: float
    upstream_token: Optional[str]
    verify_ssl: bool


def _strip_tag(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    return name.split(":")[0]


def _map_language(lang: Optional[str], default_lang: Optional[str]) -> Optional[str]:
    if lang:
        candidate = lang.strip()
    elif default_lang:
        candidate = default_lang.strip()
    else:
        return None
    lowered = candidate.lower()
    return _LANG_MAP.get(lowered, candidate)


class RivaWrapperServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        cfg: RivaConfig,
        auth_token: Optional[str],
    ):
        super().__init__(server_address, RequestHandlerClass)
        headers = {}
        if cfg.upstream_token:
            headers["Authorization"] = f"Bearer {cfg.upstream_token}"
        self.client = httpx.Client(
            base_url=cfg.base_url,
            headers=headers,
            timeout=cfg.timeout,
            verify=cfg.verify_ssl,
        )
        self.cfg = cfg
        self.auth_token = auth_token
        self.model_name = cfg.model or self._fetch_default_model()

    def _fetch_default_model(self) -> Optional[str]:
        try:
            resp = self.client.get("/v1/metadata")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("modelInfo") or []
            if models:
                return models[0].get("shortName")
        except Exception:
            return None
        return None


class OpenAICompatHandler(BaseHTTPRequestHandler):
    def _srv(self) -> RivaWrapperServer:
        return cast(RivaWrapperServer, self.server)

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
        if not self._srv().auth_token:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self._srv().auth_token}"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        server = self._srv()
        if self.path == "/health":
            self._text_response(200, "ok")
            return
        if self.path == "/v1/models":
            payload = {
                "data": [
                    {
                        "id": server.model_name or server.cfg.model or "riva-asr",
                        "object": "model",
                    }
                ]
            }
            self._json_response(200, payload)
            return
        self._text_response(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        server = self._srv()
        if self.path != "/v1/audio/transcriptions":
            self._text_response(404, "Not Found")
            return

        if not self._check_auth():
            self._text_response(401, "Unauthorized")
            return

        ctype, _ = cgi.parse_header(self.headers.get("content-type", ""))
        if ctype != "multipart/form-data":
            self._text_response(400, "Expected multipart/form-data")
            return

        if not server.model_name:
            server.model_name = server._fetch_default_model()
        elif ":" not in server.model_name:
            meta_model = server._fetch_default_model()
            if meta_model and _strip_tag(meta_model) == _strip_tag(server.model_name):
                server.model_name = meta_model

        form = cgi.FieldStorage(
            fp=cast(IO[bytes], self.rfile),
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )

        if "file" not in form:
            self._text_response(400, "Missing file field")
            return

        file_item = form["file"]
        filename = file_item.filename or "audio.wav"
        response_format = form.getvalue("response_format") or "text"
        language = _map_language(form.getvalue("language"), server.cfg.default_language)
        model = form.getvalue("model") or server.model_name
        # Normalize shorthand model names to the fully-tagged shortName exposed by metadata.
        if model and server.model_name:
            if _strip_tag(model) == _strip_tag(server.model_name):
                model = server.model_name
        prompt = form.getvalue("prompt")

        data = {}
        if language:
            data["language"] = language
        if model:
            data["model"] = model
        if prompt:
            data["prompt"] = prompt

        file_item.file.seek(0)
        files = {
            "file": (
                filename,
                file_item.file,
                file_item.type or "application/octet-stream",
            )
        }

        def _ensure_mono() -> Optional[dict]:
            try:
                file_item.file.seek(0)
                audio, sample_rate = sf.read(file_item.file, always_2d=True)
            except Exception:
                return None
            if audio.shape[1] == 1:
                return None
            mono = np.mean(audio, axis=1)
            buffer = io.BytesIO()
            sf.write(buffer, mono, sample_rate, format="WAV")
            buffer.seek(0)
            return {
                "file": (
                    os.path.splitext(filename)[0] + ".wav",
                    buffer,
                    "audio/wav",
                )
            }

        def _post(payload: dict, post_files: Optional[dict] = None) -> httpx.Response:
            return server.client.post(
                "/v1/audio/transcriptions", data=payload, files=post_files or files
            )

        try:
            upstream = _post(data)
            if upstream.status_code in {400, 404} and "model" in data:
                body = upstream.text.lower()
                if "bad model" in body or "model not found" in body:
                    retry_data = dict(data)
                    retry_data.pop("model", None)
                    upstream = _post(retry_data)
            if upstream.status_code == 500:
                body = upstream.text.lower()
                if "channel" in body and "count" in body:
                    mono_files = _ensure_mono()
                    if mono_files:
                        retry_data = dict(data)
                        retry_data.pop("model", None)
                        upstream = _post(retry_data, mono_files)
            upstream.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - surface to caller
            self._text_response(502, f"Upstream transcription failed: {exc}")
            return

        try:
            payload = upstream.json()
        except ValueError:
            payload = {"text": upstream.text}

        text = payload.get("text", "")

        if response_format == "json":
            self._json_response(200, payload)
        else:
            self._text_response(200, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default=os.getenv("WRAPPER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--listen-port", type=int, default=int(os.getenv("WRAPPER_PORT", "8099"))
    )
    parser.add_argument(
        "--riva-url", default=os.getenv("RIVA_URL", "http://127.0.0.1:9000")
    )
    parser.add_argument("--model", default=os.getenv("RIVA_MODEL"))
    parser.add_argument("--language", default=os.getenv("RIVA_LANGUAGE", "ar"))
    parser.add_argument("--auth-token", default=os.getenv("WRAPPER_AUTH_TOKEN"))
    parser.add_argument("--riva-token", default=os.getenv("RIVA_AUTH_TOKEN"))
    parser.add_argument(
        "--timeout", type=float, default=float(os.getenv("RIVA_TIMEOUT", "60"))
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for upstream Riva.",
    )
    parser.add_argument("--test-audio", help="Transcribe a file via Riva and exit.")
    parser.add_argument("--response-format", default="text", choices=["text", "json"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RivaConfig(
        base_url=args.riva_url.rstrip("/"),
        model=args.model,
        default_language=args.language,
        timeout=args.timeout,
        upstream_token=args.riva_token,
        verify_ssl=not args.insecure,
    )

    if args.test_audio:
        with open(args.test_audio, "rb") as f:
            files = {"file": (os.path.basename(args.test_audio), f)}
            data = {}
            language = _map_language(None, cfg.default_language)
            if language:
                data["language"] = language
            if cfg.model:
                data["model"] = cfg.model
            headers = {}
            if cfg.upstream_token:
                headers["Authorization"] = f"Bearer {cfg.upstream_token}"
            resp = httpx.post(
                f"{cfg.base_url}/v1/audio/transcriptions",
                data=data,
                files=files,
                headers=headers,
                timeout=cfg.timeout,
                verify=cfg.verify_ssl,
            )
        resp.raise_for_status()
        if args.response_format == "json":
            print(resp.text)
        else:
            try:
                print(resp.json().get("text", ""))
            except ValueError:
                print(resp.text)
        return 0

    server = RivaWrapperServer(
        (args.listen_host, args.listen_port), OpenAICompatHandler, cfg, args.auth_token
    )
    print(f"[INFO] Wrapper listening on http://{args.listen_host}:{args.listen_port}")
    print(f"[INFO] Riva upstream {cfg.base_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
