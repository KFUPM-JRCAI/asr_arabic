#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for Qwen3-ASR that handles JSON responses and forces Arabic language."""

from __future__ import annotations

import argparse
import cgi
import json
import os
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import IO, Optional, cast

import httpx


@dataclass
class Qwen3Config:
    base_url: str
    model: str
    timeout: float
    upstream_token: Optional[str]
    verify_ssl: bool


def _extract_text_from_qwen_response(response_text: str) -> str:
    """Extract Arabic text from Qwen3-ASR response.

    Qwen3-ASR returns: {"text":"language Arabic<asr_text>النص العربي","usage":{"type":"duration","seconds":1}}
    We need to extract only the Arabic text after <asr_text>.
    """
    try:
        # First try to parse as JSON
        data = json.loads(response_text)
        text = data.get("text", "")

        # Extract text after <asr_text> tag
        match = re.search(r"<asr_text>(.+?)(?:</asr_text>|$)", text)
        if match:
            return match.group(1).strip()

        # If no <asr_text> tag found, try to remove language prefix
        match = re.search(r"language\s+\w+\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Return the text as-is if no patterns match
        return text
    except (json.JSONDecodeError, ValueError):
        # If not JSON, return as-is
        return response_text.strip()


class Qwen3WrapperServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        cfg: Qwen3Config,
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
        self.model_name = cfg.model


class OpenAICompatHandler(BaseHTTPRequestHandler):
    def _srv(self) -> Qwen3WrapperServer:
        return cast(Qwen3WrapperServer, self.server)

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
                        "id": server.model_name,
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
        # Always force Arabic language for Qwen3-ASR
        language = "ar"
        model = form.getvalue("model") or server.model_name

        data = {
            "language": language,
            "model": model,
        }

        file_item.file.seek(0)
        files = {
            "file": (
                filename,
                file_item.file,
                file_item.type or "application/octet-stream",
            )
        }

        try:
            # Make request to Qwen3-ASR endpoint
            upstream = server.client.post(
                "/v1/audio/transcriptions", data=data, files=files
            )
            upstream.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - surface to caller
            self._text_response(502, f"Upstream transcription failed: {exc}")
            return

        # Extract text from Qwen3-ASR response
        text = _extract_text_from_qwen_response(upstream.text)

        if response_format == "json":
            self._json_response(200, {"text": text})
        else:
            self._text_response(200, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default=os.getenv("WRAPPER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--listen-port", type=int, default=int(os.getenv("WRAPPER_PORT", "8099"))
    )
    parser.add_argument(
        "--qwen3-url", default=os.getenv("QWEN3_URL", "http://127.0.0.1:8000")
    )
    parser.add_argument(
        "--model", default=os.getenv("QWEN3_MODEL", "Qwen/Qwen3-ASR-1.7B")
    )
    parser.add_argument("--auth-token", default=os.getenv("WRAPPER_AUTH_TOKEN"))
    parser.add_argument("--qwen3-token", default=os.getenv("QWEN3_AUTH_TOKEN"))
    parser.add_argument(
        "--timeout", type=float, default=float(os.getenv("QWEN3_TIMEOUT", "120"))
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for upstream Qwen3-ASR.",
    )
    parser.add_argument(
        "--test-audio", help="Transcribe a file via Qwen3-ASR and exit."
    )
    parser.add_argument("--response-format", default="text", choices=["text", "json"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Qwen3Config(
        base_url=args.qwen3_url.rstrip("/"),
        model=args.model,
        timeout=args.timeout,
        upstream_token=args.qwen3_token,
        verify_ssl=not args.insecure,
    )

    if args.test_audio:
        with open(args.test_audio, "rb") as f:
            files = {"file": (os.path.basename(args.test_audio), f)}
            data = {
                "language": "ar",
                "model": cfg.model,
            }
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
        text = _extract_text_from_qwen_response(resp.text)
        if args.response_format == "json":
            print(json.dumps({"text": text}))
        else:
            print(text)
        return 0

    server = Qwen3WrapperServer(
        (args.listen_host, args.listen_port), OpenAICompatHandler, cfg, args.auth_token
    )
    print(
        f"[INFO] Qwen3-ASR wrapper listening on http://{args.listen_host}:{args.listen_port}"
    )
    print(f"[INFO] Qwen3-ASR upstream {cfg.base_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
