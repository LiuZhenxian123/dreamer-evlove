from types import SimpleNamespace

from conftest import CLOCK, GOOD, POS_ONLY

from sea import fitness
from sea.archive import Entry
from sea.program import Program
from sea.runtime import SeaRuntime


def test_fit_budget_shares_programs_across_tasks_and_drops_the_weakest(spec, data):
    entries = {}
    for name, src in (("good", GOOD), ("clock", CLOCK), ("pos", POS_ONLY)):
        p = Program(src, spec)
        entries[name] = Entry(p, fitness.score_program(p, data))
    runtime = SimpleNamespace(max_dim=5)
    chosen = {0: [entries["good"], entries["clock"]], 1: [entries["good"], entries["pos"]]}
    pairs = SeaRuntime._fit_budget(runtime, chosen)
    by_source = {p.source: tasks for p, tasks in pairs}
    assert by_source[GOOD] == [0, 1]  # one copy, owned by both tasks
    assert sum(p.dim for p, _ in pairs) <= 5
    # good (3 dims) + pos (2 dims) fill the budget; the clock scores lowest of what still fits
    assert set(by_source) == {GOOD, POS_ONLY}
