from __future__ import annotations

from skedl.benchmarks.tasks import (
    arc_is_correct,
    build_race_prompt,
    build_commonsenseqa_prompt,
    build_piqa_prompt,
    build_arc_prompt,
    build_boolq_prompt,
    build_gsm8k_prompt,
    build_sciq_prompt,
    build_truthfulqa_prompt,
    boolq_is_correct,
    extract_arc_answer,
    extract_boolq_answer,
    extract_commonsenseqa_answer,
    extract_gsm8k_answer,
    extract_piqa_answer,
    extract_truthfulqa_answer,
    majority_vote,
    piqa_is_correct,
)


def test_extract_gsm8k_answer_prefers_final_answer_marker():
    text = "Reasoning...\nFinal answer: 43.2"
    assert extract_gsm8k_answer(text) == "43.2"


def test_extract_gsm8k_answer_falls_back_to_last_number():
    text = "Compute 50 * 0.8 = 40, then 40*1.08 = 43.2"
    assert extract_gsm8k_answer(text) == "43.2"


def test_extract_commonsenseqa_answer_parses_letter():
    text = "Let's reason. Final answer: B"
    assert extract_commonsenseqa_answer(text) == "B"


def test_majority_vote_ignores_missing_answers():
    assert majority_vote(["A", None, "B", "A", None]) == "A"


def test_prompt_builders_include_required_fields():
    gsm_prompt = build_gsm8k_prompt("How many apples?")
    csqa_prompt = build_commonsenseqa_prompt(
        question="A door is in a what?",
        choices={"A": "bank", "B": "beach", "C": "cloud", "D": "river", "E": "forest"},
    )

    assert "Final answer:" in gsm_prompt
    assert "Choices:" in csqa_prompt
    assert "A. bank" in csqa_prompt


def test_piqa_prompt_and_answer_parsing():
    prompt = build_piqa_prompt("Open a jar", "twist the lid", "freeze the jar")
    text = "Reasoning...\nFinal answer: A"

    assert "A. twist the lid" in prompt
    assert "B. freeze the jar" in prompt
    assert extract_piqa_answer(text) == "A"
    assert piqa_is_correct("A", "A")
    assert not piqa_is_correct("B", "A")


def test_boolq_prompt_and_answer_parsing():
    prompt = build_boolq_prompt("is water wet", "Water is a liquid and wets surfaces.")
    yes_text = "Short reasoning.\nFinal answer: yes"
    no_text = "Final answer: no"

    assert "Passage:" in prompt
    assert extract_boolq_answer(yes_text) == "yes"
    assert extract_boolq_answer(no_text) == "no"
    assert boolq_is_correct("yes", True)
    assert boolq_is_correct("no", False)
    assert not boolq_is_correct("yes", False)


def test_arc_prompt_and_answer_parsing_reuses_letter_parser():
    prompt = build_arc_prompt(
        "Which is a mammal?",
        {"A": "shark", "B": "whale", "C": "trout", "D": "eel"},
    )

    assert "Choices:" in prompt
    assert "B. whale" in prompt
    assert extract_arc_answer("Final answer: B") == "B"
    assert arc_is_correct("B", "B")
    assert not arc_is_correct("A", "B")


def test_sciq_and_race_prompt_builders_include_context_and_choices():
    sciq_prompt = build_sciq_prompt(
        question="Who proposed the theory of evolution by natural selection?",
        support="",
        choices={"A": "darwin", "B": "Linnaeus", "C": "shaw", "D": "Scopes"},
    )
    race_prompt = build_race_prompt(
        article="A short article about studying.",
        question="What did Timothy feel?",
        choices={"A": "Happy", "B": "Angry", "C": "Tired", "D": "Excited"},
    )

    assert "Final answer:" in sciq_prompt
    assert "A. darwin" in sciq_prompt
    assert "Passage:" in race_prompt
    assert "Question:" in race_prompt
    assert "D. Excited" in race_prompt


def test_truthfulqa_prompt_and_answer_extraction():
    prompt = build_truthfulqa_prompt("What happens if you eat watermelon seeds?")
    text = "Brief reasoning.\nFinal answer: Nothing happens."

    assert "Answer the question truthfully" in prompt
    assert extract_truthfulqa_answer(text) == "Nothing happens."
    assert extract_truthfulqa_answer("No marker present") == "No marker present"
