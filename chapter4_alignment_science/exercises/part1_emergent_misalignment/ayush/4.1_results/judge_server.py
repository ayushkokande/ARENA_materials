"""OpenAI-style /v1/chat/completions endpoint backed by `claude -p` (Claude Code headless, Haiku).

Lets the ARENA 4.1 notebook (running on a remote pod, reached through an SSH reverse tunnel) use
Claude as its LLM judge via the user's Claude Code login, instead of an OpenRouter key.
Every request/response is appended to judge_log.jsonl next to this file.
"""

import json
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8765
MAX_PARALLEL = 8
MODEL = "haiku"
DEFAULT_SYSTEM = "You are a helpful assistant. Follow the user's instructions exactly."
LOG = Path(__file__).with_name("judge_log.jsonl")

_sem = threading.Semaphore(MAX_PARALLEL)
_log_lock = threading.Lock()
_stats = {"calls": 0, "errors": 0, "in_tokens": 0, "out_tokens": 0}


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content-part format
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def call_claude(messages: list[dict]) -> tuple[str, dict]:
    system_parts = [_content_to_text(m["content"]) for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]
    if len(convo) == 1:
        prompt = _content_to_text(convo[0]["content"])
    else:  # flatten multi-turn into one transcript
        prompt = "\n\n".join(f"[{m['role'].upper()}]\n{_content_to_text(m['content'])}" for m in convo)
        prompt += "\n\n[ASSISTANT]\n"
    system = "\n\n".join(system_parts) or DEFAULT_SYSTEM
    cmd = [
        "claude", "-p", "--model", MODEL, "--output-format", "json",
        "--system-prompt", system, "--tools", "", "--strict-mcp-config",
        "--no-session-persistence", "--setting-sources", "",
    ]
    with _sem:
        for attempt in range(3):
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=180)
            try:
                data = json.loads(proc.stdout)
                if not data.get("is_error"):
                    return data.get("result", ""), data.get("usage", {})
            except json.JSONDecodeError:
                pass
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"claude -p failed: rc={proc.returncode} out={proc.stdout[:300]} err={proc.stderr[:300]}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default stderr logging
        pass

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, **_stats})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": f"unsupported path {self.path}"}})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        t0 = time.time()
        try:
            text, usage = call_claude(req.get("messages", []))
        except Exception as e:  # surface as an API error so the OpenAI client raises
            _stats["errors"] += 1
            return self._send(500, {"error": {"message": str(e)}})
        in_tok = sum(usage.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        out_tok = usage.get("output_tokens", 0)
        _stats["calls"] += 1
        _stats["in_tokens"] += in_tok
        _stats["out_tokens"] += out_tok
        with _log_lock, LOG.open("a") as f:
            f.write(json.dumps({"t": t0, "secs": round(time.time() - t0, 2), "requested_model": req.get("model"),
                                "messages": req.get("messages"), "response": text}) + "\n")
        self._send(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(t0),
            "model": f"claude-{MODEL}",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok},
        })


if __name__ == "__main__":
    print(f"judge server on 127.0.0.1:{PORT} (claude -p --model {MODEL}, max {MAX_PARALLEL} parallel)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
