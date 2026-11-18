import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import torch
from conftest import CLOCK, GOOD, NOISE, POS_ONLY, make_data
from omegaconf import OmegaConf

from sea.evolve import Evolver
from sea.library import Library
from sea.llm import ChatLLM, ScriptedLLM, code_blocks
from sea.program import Program

CFG = OmegaConf.create({
    "capacity": 16, "per_call": 2, "calls_per_gen": 2, "generations": 2, "max_program_dim": 16,
    "weights": {"closure": 1.0, "reward": 1.0, "visible": 1.0, "control": 0.25},
    "modes": {"mutate": 0.5, "crossover": 0.25, "explore": 0.25},
})  # fmt: skip


def block(*sources):
    return "Here you go.\n\n" + "\n\n".join(f"```python\n{s}```" for s in sources)


def test_code_blocks():
    assert code_blocks(block(GOOD, CLOCK)) == [GOOD.strip(), CLOCK.strip()]
    assert code_blocks("no code here") == []


def test_evolver_improves_and_feeds_back_failures(spec, data, tmp_path):
    broken = "def phi(s):\n    return {'x': s['missing']}\n"
    llm = ScriptedLLM([block(CLOCK, NOISE), block(broken, POS_ONLY), block(GOOD), "I refuse."])
    evolver = Evolver(spec, "reach the goal", llm, CFG, logdir=tmp_path, name="toy")
    archive = evolver.run(data)
    assert archive.ranked()[0].program.source == GOOD
    assert evolver.generation == 2 and llm.calls == 4
    # The rejected program's error is shown to the LLM in a later prompt.
    assert any("unknown privileged fields" in m[1]["content"] for m in llm.prompts[2:])
    # Later prompts show scored parents; the first one cannot.
    assert "Best abstractions so far" not in llm.prompts[0][1]["content"]
    assert "score" in llm.prompts[-1][1]["content"]
    history = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [h["generation"] for h in history] == [1, 2] and history[-1]["best"] >= history[0]["best"]
    assert (tmp_path / "archive.json").exists()


def test_world_model_diagnostics_reach_the_prompt(spec, data):
    llm = ScriptedLLM([block(GOOD)])
    Evolver(spec, "reach", llm, CFG, name="toy").run(data, generations=1, diagnostics={"goal_dist": 0.93})
    assert "goal_dist: 0.93" in llm.prompts[0][1]["content"]


def test_library_reuse_is_verified_on_the_new_task(spec, data, tmp_path):
    lib = Library(tmp_path / "lib")
    first = Evolver(spec, "reach", ScriptedLLM([block(GOOD, NOISE, CLOCK)]), CFG, library=lib, name="task_a")
    archive = first.run(data, generations=1)
    picked = archive.select_active(data, 16)
    info = lib.absorb("task_a", archive, data, picked)
    assert info["added"] == len(archive) and lib.entries[0]["tasks"]["task_a"]["selected"]

    # A second task gets the library's programs re-scored on its own data before any LLM call.
    other = make_data(spec, seed=7)
    second = Evolver(
        spec, "reach", ScriptedLLM([block(POS_ONLY)]), CFG, library=Library(tmp_path / "lib"), name="task_b"
    )
    second.seed_from_library(other)
    assert second.llm.calls == 0 and second.stats["library_kept"] >= 1
    assert second.archive.ranked()[0].origin == "library" and second.archive.ranked()[0].program.source == GOOD
    archive_b = second.run(other, generations=1)
    info = second.library.absorb("task_b", archive_b, other, archive_b.select_active(other, 16))
    assert info["reused"] >= 1
    stats = second.library.reuse_stats()
    assert stats["used_on_multiple_tasks"] >= 1 and stats["selected_on_multiple_tasks"] >= 1


def test_library_merges_equivalent_programs(spec, data, tmp_path):
    lib = Library(tmp_path / "lib")
    same = 'def phi(s):\n    d = s["goal"] - s["pos"]\n    return {"offset": 5.0 * d, "gap": norm(d)}\n'
    for name, src in (("a", GOOD), ("b", same)):
        ev = Evolver(spec, "reach", ScriptedLLM([block(src)]), CFG, name=name)
        lib.absorb(name, ev.run(data, generations=1), data)
    assert len(lib.entries) == 1 and set(lib.entries[0]["tasks"]) == {"a", "b"}


def test_refactor_accepts_only_equivalent_rewrites(spec, data, tmp_path):
    lib = Library(tmp_path / "lib")
    sources = [
        'def phi(s):\n    return {"gap": norm(s["goal"] - s["pos"])}\n',
        'def phi(s):\n    return {"gap": norm(s["goal"] - s["pos"]), "t": s["clock"]}\n',
        'def phi(s):\n    return {"p": s["pos"]}\n',
    ]
    ev = Evolver(spec, "reach", ScriptedLLM([block(*sources)]), CFG, name="a")
    lib.absorb("a", ev.run(data, generations=1), data)
    ids = {e["source"]: e["id"] for e in lib.entries}
    helper = 'def goal_gap(s):\n    """Distance from agent to goal."""\n    return norm(s["goal"] - s["pos"])\n'
    good0 = f'# id: {ids[sources[0]]}\ndef phi(s):\n    return {{"gap": goal_gap(s)}}\n'
    good1 = f'# id: {ids[sources[1]]}\ndef phi(s):\n    return {{"gap": goal_gap(s), "t": s["clock"]}}\n'
    wrong = f'# id: {ids[sources[2]]}\ndef phi(s):\n    return {{"p": s["pos"] + goal_gap(s)}}\n'

    assert lib.refactor(ScriptedLLM([block(helper, good0, good1, wrong)]), data) == {"helpers_added": 1, "rewritten": 2}
    by_id = {e["id"]: e["source"] for e in lib.entries}
    assert "goal_gap(s)" in by_id[ids[sources[0]]] and by_id[ids[sources[2]]] == sources[2]
    assert "goal_gap(s): Distance from agent to goal." in lib.helper_docs()
    # Reloaded from disk, the rewritten programs still compile against the stored helper.
    again = Library(tmp_path / "lib")
    p = Program(by_id[ids[sources[0]]], spec, again.helpers)
    assert torch.allclose(p(data.priv), Program(sources[0], spec)(data.priv), atol=1e-5)

    # A helper nobody can use correctly is not adopted.
    lib2 = Library(tmp_path / "lib2")
    lib2.absorb("a", ev.archive, data)
    assert lib2.refactor(ScriptedLLM([block(helper, wrong)]), data)["helpers_added"] == 0 and not lib2.helpers


def test_insights_are_stored(spec, data, tmp_path):
    lib = Library(tmp_path / "lib")
    ev = Evolver(spec, "reach", ScriptedLLM([block(GOOD, NOISE, CLOCK, POS_ONLY)]), CFG, name="a")
    lib.absorb("a", ev.run(data, generations=1), data)
    text = lib.update_insights(
        ScriptedLLM(["Sure:\n- Relative offsets to the goal score well.\n- Raw noise fails.\nBye"])
    )
    assert text.splitlines() == ["- Relative offsets to the goal score well.", "- Raw noise fails."]
    assert Library(tmp_path / "lib").insights == text


@pytest.fixture
def chat_server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body))
            reply = {
                "choices": [{"message": {"content": block(GOOD)}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()


def test_chat_llm_over_http_with_cache(chat_server, tmp_path, monkeypatch):
    url, seen = chat_server
    monkeypatch.setenv("SEA_LLM_BASE_URL", url)
    monkeypatch.setenv("SEA_LLM_API_KEY", "test-key")
    llm = ChatLLM("some-model", logdir=tmp_path)
    messages = [{"role": "user", "content": "hi"}]
    assert code_blocks(llm.chat(messages, tag="t")) == [GOOD.strip()]
    assert (
        seen[0][0] == "/v1/chat/completions" and seen[0][1] == "Bearer test-key" and seen[0][2]["model"] == "some-model"
    )
    llm.chat(messages, tag="t")  # served from the cache
    llm.chat(messages, tag="t", index=1)  # a distinct sample is a new request
    assert len(seen) == 2 and llm.calls == 2 and llm.usage == {"prompt_tokens": 22, "completion_tokens": 14}
    log = (tmp_path / "llm_log.jsonl").read_text()
    assert "test-key" not in log and len(log.splitlines()) == 2
