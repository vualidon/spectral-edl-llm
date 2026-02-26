from __future__ import annotations

import re
from collections import Counter
from hashlib import md5


_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*([^\n]+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
_GSM8K_HASH_RE = re.compile(r"####\s*([^\n]+)")
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def build_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math word problem. Reason step by step.\n"
        "At the end, write exactly `Final answer: <number>` on its own line.\n\n"
        f"Question: {question}\n"
    )


def build_commonsenseqa_prompt(question: str, choices: dict[str, str]) -> str:
    ordered = "\n".join(f"{label}. {choices[label]}" for label in ["A", "B", "C", "D", "E"] if label in choices)
    return (
        "Answer the multiple-choice commonsense question. Think step by step briefly.\n"
        "At the end, write exactly `Final answer: <A/B/C/D/E>` on its own line.\n\n"
        f"Question: {question}\n"
        f"Choices:\n{ordered}\n"
    )


def build_piqa_prompt(goal: str, sol1: str, sol2: str) -> str:
    return (
        "Choose the better solution for the physical commonsense goal. Think step by step briefly.\n"
        "At the end, write exactly `Final answer: <A/B>` on its own line.\n\n"
        f"Goal: {goal}\n"
        f"Choices:\nA. {sol1}\nB. {sol2}\n"
    )


def build_boolq_prompt(question: str, passage: str) -> str:
    return (
        "Answer the question based on the passage. Think step by step briefly.\n"
        "At the end, write exactly `Final answer: <yes/no>` on its own line.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
    )


def build_arc_prompt(question: str, choices: dict[str, str]) -> str:
    ordered = "\n".join(f"{label}. {text}" for label, text in choices.items())
    return (
        "Answer the science multiple-choice question. Think step by step briefly.\n"
        "At the end, write exactly `Final answer: <A/B/C/D/E>` on its own line.\n\n"
        f"Question: {question}\n"
        f"Choices:\n{ordered}\n"
    )


def build_sciq_prompt(question: str, support: str, choices: dict[str, str]) -> str:
    ordered = "\n".join(f"{label}. {text}" for label, text in choices.items())
    support_block = f"Support: {support}\n" if support and str(support).strip() else ""
    return (
        "Answer the science question using the provided context if useful. Think step by step briefly.\n"
        "At the end, write exactly `Final answer: <A/B/C/D>` on its own line.\n\n"
        f"{support_block}"
        f"Question: {question}\n"
        f"Choices:\n{ordered}\n"
    )


def build_race_prompt(article: str, question: str, choices: dict[str, str]) -> str:
    ordered = "\n".join(f"{label}. {text}" for label, text in choices.items())
    return (
        "Read the passage and answer the multiple-choice question. Think step by step briefly.\n"
        "At the end, write exactly `Final answer: <A/B/C/D>` on its own line.\n\n"
        f"Passage: {article}\n"
        f"Question: {question}\n"
        f"Choices:\n{ordered}\n"
    )


def build_truthfulqa_prompt(question: str) -> str:
    return (
        "Answer the question truthfully and concisely. Think step by step briefly.\n"
        "At the end, write exactly `Final answer: <answer>` on its own line.\n\n"
        f"Question: {question}\n"
    )


def build_sciq_choice_map(
    *,
    question: str,
    correct_answer: str,
    distractors: list[str],
) -> tuple[dict[str, str], str]:
    # Deterministic placement avoids leaking the correct answer position while keeping runs reproducible.
    labels = ["A", "B", "C", "D"]
    answers = [str(correct_answer)] + [str(x) for x in distractors[:3]]
    seed_bytes = md5(str(question).encode("utf-8")).digest()
    insert_idx = int(seed_bytes[0]) % len(labels)
    choice_texts = [None, None, None, None]
    choice_texts[insert_idx] = answers[0]
    d_idx = 1
    for i in range(4):
        if choice_texts[i] is None:
            choice_texts[i] = answers[d_idx]
            d_idx += 1
    choice_map = {label: str(text) for label, text in zip(labels, choice_texts)}
    return choice_map, labels[insert_idx]


def _normalize_number_token(text: str) -> str | None:
    if text is None:
        return None
    token = text.strip().replace("$", "").replace(",", "")
    if not token:
        return None
    # trim trailing punctuation often produced by generation
    token = token.rstrip(" .,)(")
    try:
        value = float(token)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return format(value, "g")


def extract_gsm8k_gold_answer(answer: str) -> str | None:
    m = _GSM8K_HASH_RE.search(answer)
    if m:
        return _normalize_number_token(m.group(1))
    matches = _NUMBER_RE.findall(answer or "")
    if not matches:
        return None
    return _normalize_number_token(matches[-1])


def extract_gsm8k_answer(text: str) -> str | None:
    if not text:
        return None
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        candidate = _normalize_number_token(m.group(1))
        if candidate is not None:
            return candidate
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return _normalize_number_token(matches[-1])


def extract_commonsenseqa_answer(text: str) -> str | None:
    if not text:
        return None
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        letter = _LETTER_RE.search(m.group(1))
        if letter:
            return letter.group(1).upper()
    letters = _LETTER_RE.findall(text)
    if not letters:
        return None
    return letters[-1].upper()


def extract_piqa_answer(text: str) -> str | None:
    if not text:
        return None
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        letter = re.search(r"\b([AB])\b", m.group(1), re.IGNORECASE)
        if letter:
            return letter.group(1).upper()
    letters = re.findall(r"\b([AB])\b", text, re.IGNORECASE)
    if not letters:
        return None
    return letters[-1].upper()


def extract_boolq_answer(text: str) -> str | None:
    if not text:
        return None
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        yn = _YESNO_RE.search(m.group(1))
        if yn:
            return yn.group(1).lower()
    matches = _YESNO_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower()


def extract_arc_answer(text: str) -> str | None:
    return extract_commonsenseqa_answer(text)


def extract_truthfulqa_answer(text: str) -> str | None:
    if not text:
        return None
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    return stripped or None


def majority_vote(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a]
    if not valid:
        return None
    counts = Counter(valid)
    # deterministic tie-break by count then lexical order
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def gsm8k_is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def csqa_is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    return pred.upper() == gold.upper()


def piqa_is_correct(pred: str | None, gold: str | None) -> bool:
    return csqa_is_correct(pred, gold)


def arc_is_correct(pred: str | None, gold: str | None) -> bool:
    return csqa_is_correct(pred, gold)


def boolq_is_correct(pred: str | None, gold: bool | int | str | None) -> bool:
    if pred is None or gold is None:
        return False
    if isinstance(gold, str):
        gold_norm = gold.strip().lower() in {"true", "yes", "1"}
    else:
        gold_norm = bool(gold)
    return pred.strip().lower() == ("yes" if gold_norm else "no")


def normalize_mcq_label(label: str | None) -> str | None:
    if label is None:
        return None
    text = str(label).strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    digit_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return digit_map.get(text)
