from __future__ import annotations

import io


def test_truthfulqa_reference_selection_and_threshold_labeling():
    from skedl.benchmarks.truthfulqa import references_from_row, score_truthfulqa_prediction

    row = {
        "best_answer": "The seeds pass through your digestive system.",
        "correct_answers": ["Nothing happens.", "The seeds pass through your digestive system."],
    }

    refs_best = references_from_row(row, mode="best")
    refs_max = references_from_row(row, mode="max_correct")
    assert refs_best == ["The seeds pass through your digestive system."]
    assert "Nothing happens." in refs_max

    class FakeScorer:
        name = "fake"

        def score(self, prediction: str, references: list[str]) -> float:
            return 0.6 if "Nothing" in prediction else 0.2

    scored = score_truthfulqa_prediction(
        prediction="Nothing happens.",
        row=row,
        scorer=FakeScorer(),
        threshold=0.5,
        reference_mode="max_correct",
    )
    assert scored["label"] is True
    assert scored["score"] == 0.6
    assert scored["scorer"] == "fake"


def test_bleurt_scorer_uses_sidecar_python_from_env(monkeypatch):
    import json
    from pathlib import Path

    from skedl.benchmarks import truthfulqa as tqa

    calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(json.dumps({"scores": [0.2, 0.8]}) + "\n")
            self.stderr = io.StringIO()
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    def _fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc()

    monkeypatch.setenv("SKEDL_BLEURT_SIDECAR_PYTHON", "/tmp/fake-python")
    monkeypatch.setattr(tqa, "_ensure_bleurt_checkpoint", lambda *args, **kwargs: "/tmp/BLEURT-20")
    monkeypatch.setattr(tqa.subprocess, "Popen", _fake_popen)

    scorer = tqa.make_truthfulqa_scorer("bleurt")
    got = scorer.score("prediction", ["ref1", "ref2"])

    assert got == 0.8
    assert calls, "sidecar process should be started"
    assert calls[0][0] == "/tmp/fake-python"
    assert Path(calls[0][1]).name == "bleurt_sidecar.py"
