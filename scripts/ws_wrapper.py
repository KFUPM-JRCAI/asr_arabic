#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for a whisper-live WebSocket server."""

from __future__ import annotations

import argparse
import cgi
import json
import os
import tempfile
import time
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
import uuid
import shutil

import av
import numpy as np
import websocket

END_OF_AUDIO = "END_OF_AUDIO"


@dataclass
class WSConfig:
    host: str
    port: int
    use_wss: bool
    model: str
    language: Optional[str]
    task: str
    use_vad: bool
    max_clients: int
    max_connection_time: int
    send_last_n_segments: int
    no_speech_thresh: float
    clip_audio: bool
    same_output_threshold: int
    chunk_frames: int
    connect_timeout: float
    read_timeout: float
    initial_wait: float
    final_wait: float
    debug: bool
    pace: float


def _resample_to_temp(audio_path: Path, sr: int = 16000) -> Path:
    container = av.open(str(audio_path))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = Path(tmp.name)
    tmp.close()

    output_container = av.open(str(tmp_path), mode="w")
    output_stream = output_container.add_stream("pcm_s16le", rate=sr)
    output_stream.layout = "mono"

    try:
        for frame in container.decode(audio=0):
            frame.pts = None
            resampled_frames = resampler.resample(frame)
            if resampled_frames is None:
                continue
            for resampled_frame in resampled_frames:
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)
        for packet in output_stream.encode(None):
            output_container.mux(packet)
    finally:
        output_container.close()
        container.close()

    return tmp_path


def _float32_bytes_from_pcm16(pcm_bytes: bytes) -> bytes:
    raw = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (raw.astype(np.float32) / 32768.0).tobytes()


def _recv_json(ws: websocket.WebSocket) -> Optional[dict]:
    raw = ws.recv()
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def transcribe_via_ws(audio_path: Path, cfg: WSConfig, language: Optional[str], model: Optional[str]) -> str:
    socket_protocol = "wss" if cfg.use_wss else "ws"
    socket_url = f"{socket_protocol}://{cfg.host}:{cfg.port}"
    ws = websocket.create_connection(socket_url, timeout=cfg.connect_timeout)
    ws.settimeout(cfg.read_timeout)

    try:
        ws.send(
            json.dumps(
                {
                    "uid": str(uuid.uuid4()),
                    "language": language or cfg.language,
                    "task": cfg.task,
                    "model": model or cfg.model,
                    "use_vad": cfg.use_vad,
                    "max_clients": cfg.max_clients,
                    "max_connection_time": cfg.max_connection_time,
                    "send_last_n_segments": cfg.send_last_n_segments,
                    "no_speech_thresh": cfg.no_speech_thresh,
                    "clip_audio": cfg.clip_audio,
                    "same_output_threshold": cfg.same_output_threshold,
                }
            )
        )

        while True:
            msg = _recv_json(ws)
            if msg is None:
                raise RuntimeError("No response from WebSocket server during handshake.")
            if cfg.debug:
                print(f"[DEBUG] handshake: {msg}")
            if msg.get("status") == "ERROR":
                raise RuntimeError(msg.get("message", "Server error during handshake."))
            if msg.get("message") == "SERVER_READY":
                break

        resampled = _resample_to_temp(audio_path)
        try:
            chunk_duration = cfg.chunk_frames / 16000.0
            with wave.open(str(resampled), "rb") as wavfile:
                while True:
                    data = wavfile.readframes(cfg.chunk_frames)
                    if data == b"":
                        break
                    ws.send(_float32_bytes_from_pcm16(data), opcode=websocket.ABNF.OPCODE_BINARY)
                    if cfg.pace > 0:
                        time.sleep(chunk_duration / cfg.pace)
            ws.send(END_OF_AUDIO.encode("utf-8"), opcode=websocket.ABNF.OPCODE_BINARY)
        finally:
            try:
                resampled.unlink()
            except OSError:
                pass

        segments = []
        last_end = -1.0
        last_segment = None
        last_msg_time = time.time()
        start_wait = last_msg_time
        seen_segment = False

        while True:
            try:
                msg = _recv_json(ws)
            except websocket.WebSocketTimeoutException:
                now = time.time()
                if not seen_segment:
                    if now - start_wait > cfg.initial_wait:
                        break
                elif now - last_msg_time > cfg.final_wait:
                    break
                continue
            if msg is None:
                break
            if cfg.debug:
                print(f"[DEBUG] message: {msg}")
            last_msg_time = time.time()

            if msg.get("status") == "ERROR":
                raise RuntimeError(msg.get("message", "Server error during transcription."))
            if msg.get("message") == "DISCONNECT":
                break
            if "segments" not in msg:
                continue

            for seg in msg["segments"]:
                if seg.get("completed"):
                    try:
                        end = float(seg.get("end", -1))
                    except (TypeError, ValueError):
                        end = -1
                    if end > last_end:
                        segments.append(seg)
                        last_end = end
                        seen_segment = True
                else:
                    last_segment = seg

        if last_segment and (not segments or last_segment.get("text") != segments[-1].get("text")):
            segments.append(last_segment)

        return "".join(seg.get("text", "") for seg in segments).strip()
    finally:
        try:
            ws.close()
        except Exception:
            pass


class WrapperServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, cfg: WSConfig, auth_token: Optional[str]):
        super().__init__(server_address, RequestHandlerClass)
        self.cfg = cfg
        self.auth_token = auth_token


class OpenAICompatHandler(BaseHTTPRequestHandler):
    server: WrapperServer

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
        if not self.server.auth_token:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.server.auth_token}"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        if self.path == "/health":
            self._text_response(200, "ok")
            return
        if self.path == "/v1/models":
            payload = {"data": [{"id": self.server.cfg.model, "object": "model"}]}
            self._json_response(200, payload)
            return
        self._text_response(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
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
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )

        if "file" not in form:
            self._text_response(400, "Missing file field")
            return

        file_item = form["file"]
        suffix = Path(file_item.filename or "audio.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            shutil.copyfileobj(file_item.file, tmp)
            tmp_path = Path(tmp.name)
        finally:
            tmp.close()

        response_format = form.getvalue("response_format") or "text"
        model = form.getvalue("model")
        language = form.getvalue("language")

        try:
            text = transcribe_via_ws(tmp_path, self.server.cfg, language, model)
        except Exception as exc:  # noqa: BLE001 - surfaces error to client
            self._text_response(500, f"Transcription failed: {exc}")
            return
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        if response_format == "json":
            self._json_response(200, {"text": text})
        else:
            self._text_response(200, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default=os.getenv("WRAPPER_HOST", "127.0.0.1"))
    parser.add_argument("--listen-port", type=int, default=int(os.getenv("WRAPPER_PORT", "8099")))
    parser.add_argument("--ws-host", default=os.getenv("WS_BACKEND_HOST", "127.0.0.1"))
    parser.add_argument("--ws-port", type=int, default=int(os.getenv("WS_BACKEND_PORT", "20168")))
    parser.add_argument("--use-wss", action="store_true", default=os.getenv("WS_BACKEND_WSS") == "1")
    parser.add_argument("--model", default=os.getenv("WS_MODEL", "small"))
    parser.add_argument("--language", default=os.getenv("WS_LANGUAGE"))
    parser.add_argument("--task", default=os.getenv("WS_TASK", "transcribe"))
    parser.add_argument("--use-vad", action="store_true", default=os.getenv("WS_USE_VAD", "1") == "1")
    parser.add_argument("--max-clients", type=int, default=int(os.getenv("WS_MAX_CLIENTS", "4")))
    parser.add_argument("--max-connection-time", type=int, default=int(os.getenv("WS_MAX_CONNECTION_TIME", "600")))
    parser.add_argument("--send-last-n", type=int, default=int(os.getenv("WS_SEND_LAST_N", "1000")))
    parser.add_argument("--no-speech-thresh", type=float, default=float(os.getenv("WS_NO_SPEECH_THRESH", "0.45")))
    parser.add_argument("--clip-audio", action="store_true", default=os.getenv("WS_CLIP_AUDIO", "0") == "1")
    parser.add_argument("--same-output-threshold", type=int, default=int(os.getenv("WS_SAME_OUTPUT_THRESHOLD", "10")))
    parser.add_argument("--chunk-frames", type=int, default=int(os.getenv("WS_CHUNK_FRAMES", "4096")))
    parser.add_argument("--connect-timeout", type=float, default=float(os.getenv("WS_CONNECT_TIMEOUT", "10")))
    parser.add_argument("--read-timeout", type=float, default=float(os.getenv("WS_READ_TIMEOUT", "1.5")))
    parser.add_argument("--initial-wait", type=float, default=float(os.getenv("WS_INITIAL_WAIT", "60")))
    parser.add_argument("--final-wait", type=float, default=float(os.getenv("WS_FINAL_WAIT", "15")))
    parser.add_argument("--pace", type=float, default=float(os.getenv("WS_PACE", "1.0")))
    parser.add_argument("--auth-token", default=os.getenv("WRAPPER_AUTH_TOKEN"))
    parser.add_argument("--debug", action="store_true", default=os.getenv("WRAPPER_DEBUG") == "1")
    parser.add_argument("--test-audio", help="Transcribe a file via WebSocket and exit.")
    parser.add_argument("--response-format", default="text", choices=["text", "json"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = WSConfig(
        host=args.ws_host,
        port=args.ws_port,
        use_wss=args.use_wss,
        model=args.model,
        language=args.language,
        task=args.task,
        use_vad=args.use_vad,
        max_clients=args.max_clients,
        max_connection_time=args.max_connection_time,
        send_last_n_segments=args.send_last_n,
        no_speech_thresh=args.no_speech_thresh,
        clip_audio=args.clip_audio,
        same_output_threshold=args.same_output_threshold,
        chunk_frames=args.chunk_frames,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        initial_wait=args.initial_wait,
        final_wait=args.final_wait,
        debug=args.debug,
        pace=args.pace,
    )

    if args.test_audio:
        text = transcribe_via_ws(Path(args.test_audio), cfg, args.language, args.model)
        if args.response_format == "json":
            print(json.dumps({"text": text}))
        else:
            print(text)
        return 0

    server = WrapperServer((args.listen_host, args.listen_port), OpenAICompatHandler, cfg, args.auth_token)
    print(f"[INFO] Wrapper listening on http://{args.listen_host}:{args.listen_port}")
    print(f"[INFO] WS backend ws://{cfg.host}:{cfg.port} (wss={cfg.use_wss})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
