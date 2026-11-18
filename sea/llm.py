"""LLM access through any OpenAI-compatible chat-completions endpoint.

Configure with environment variables so no credentials live in configs or logs:

    SEA_LLM_BASE_URL   e.g. https://api.openai.com/v1  or  http://localhost:8000/v1 (vLLM)
    SEA_LLM_API_KEY    bearer token (may be empty for local servers)

Every exchange is appended to a JSONL log, and responses are cached on disk by request
hash so a rerun with the same prompts and sampling index replays instead of re-querying.
"""

import hashlib
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.request


class ChatLLM:
    def __init__(self, model, temperature=1.0, max_tokens=4096, logdir=None, timeout=300, retries=5, extra=None):
        self.model = str(model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout, self.retries = int(timeout), int(retries)
        self.extra = dict(extra or {})
        self.base_url = os.environ.get("SEA_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._key = os.environ.get("SEA_LLM_API_KEY", "")
        self.logdir = None if logdir is None else pathlib.Path(logdir)
        if self.logdir is not None:
            (self.logdir / "cache").mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _request(self, payload):
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
                if e.code in (400, 401, 403, 404):
                    break  # not retryable
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last = repr(e)
            time.sleep(min(60, 2**attempt * 2))
        raise RuntimeError(f"LLM request to {self.base_url} failed: {last}")

    def chat(self, messages, tag="", index=0):
        """Returns the assistant text. `index` distinguishes repeated samples of one prompt."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra,
        }
        key = hashlib.sha1(json.dumps([payload, index], sort_keys=True).encode()).hexdigest()
        cache = None if self.logdir is None else self.logdir / "cache" / f"{key}.json"
        if cache is not None and cache.exists():
            return json.loads(cache.read_text())["text"]
        body = self._request(payload)
        text = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
        self.calls += 1
        for k in self.usage:
            self.usage[k] += int(usage.get(k, 0) or 0)
        if self.logdir is not None:
            cache.write_text(json.dumps({"text": text}))
            with (self.logdir / "llm_log.jsonl").open("a") as f:
                record = {"time": time.time(), "tag": tag, "model": self.model, "messages": messages, "text": text}
                f.write(json.dumps({**record, "usage": usage}) + "\n")
        return text


class ScriptedLLM:
    """Deterministic stand-in used by tests and dry runs: replays canned responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.prompts = []

    def chat(self, messages, tag="", index=0):
        self.prompts.append(messages)
        text = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return text(messages) if callable(text) else text


_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def code_blocks(text):
    """Python code blocks in an LLM response, in order."""
    return [b.strip() for b in _BLOCK.findall(text) if b.strip()]


def make_llm(cfg, logdir=None):
    return ChatLLM(
        cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        logdir=logdir,
        extra=dict(cfg.get("extra", {}) or {}),
    )
