"""Tests for the core analysis engine in linter.py.

Pure metric helpers are tested directly; NLP-dependent checks use a real spaCy
doc (matching the style of test_suppressions.py), since a model is installed.
"""
import pytest

import linter
from linter import (
    calculate_reading_time,
    calculate_dialogue_ratio,
    count_syllables,
    calculate_readability,
    calculate_prose_score,
    calculate_drone_factor,
    get_sentences,
    find_sentence_id_fast,
    find_sentence_id_fast_char,
    get_redundancy_analysis,
    get_possessive_errors_analysis,
    get_spelling_analysis,
    lint_prose,
)
from text_refiner import get_spacy_doc


# --------------------------------------------------------------------------- #
# Pure metric helpers (no NLP required)
# --------------------------------------------------------------------------- #

def test_reading_time_empty():
    assert calculate_reading_time("", word_count=0) == "0s"


def test_reading_time_scales_with_length():
    # 450 words at 225 wpm -> ~2 minutes
    rt = calculate_reading_time("word " * 450, word_count=450)
    assert rt.startswith("2m")


def test_count_syllables():
    assert count_syllables("cat") == 1
    assert count_syllables("hello") == 2
    assert count_syllables("") == 0
    # silent trailing 'e' is dropped, but 'le' is kept
    assert count_syllables("make") == 1
    assert count_syllables("simple") == 2


def test_get_sentences_splits_on_punctuation():
    sents = get_sentences("Hello there. How are you? I am fine!")
    assert sents == ["Hello there.", "How are you?", "I am fine!"]


def test_get_sentences_ignores_blank():
    assert get_sentences("   ") == []


def test_find_sentence_id_fast():
    starts = [0, 10, 25]  # token indices where sentences begin
    assert find_sentence_id_fast(0, starts) == 1
    assert find_sentence_id_fast(12, starts) == 2
    assert find_sentence_id_fast(30, starts) == 3


def test_find_sentence_id_fast_char():
    starts = [0, 20, 40]
    assert find_sentence_id_fast_char(5, starts) == 1
    assert find_sentence_id_fast_char(25, starts) == 2


def test_readability_returns_scores():
    text = "The quick brown fox jumps over the lazy dog. It was a sunny day."
    sentences = get_sentences(text)
    r = calculate_readability(text, sentences)
    assert r is not None
    assert "score" in r and "grade_level" in r
    assert r["metrics"]["sentence_count"] == 2


def test_readability_none_on_empty():
    assert calculate_readability("", []) is None


def test_drone_factor_uniform_is_low():
    # Identical sentence lengths -> robotic -> low score
    uniform = ["one two three four five"] * 6
    varied = ["short.", "a slightly longer sentence here now",
              "tiny", "this one rambles on and on with many words indeed yes",
              "mid length sentence", "x"]
    assert calculate_drone_factor(uniform) < calculate_drone_factor(varied)


def test_drone_factor_empty_default():
    assert calculate_drone_factor([]) == 5.0


def test_prose_score_perfect_clean_report():
    assert calculate_prose_score({}) == 100


def test_prose_score_penalizes_spelling():
    report = {"spelling": [{"word": "teh"}, {"word": "recieve"}]}
    assert calculate_prose_score(report) < 100


def test_prose_score_stays_in_range_when_heavily_penalized():
    # Each penalty category is individually capped, so even a very poor report
    # stays within [0, 100] and well below a clean score.
    report = {
        "spelling": [{"word": f"w{i}"} for i in range(50)],
        "executive_summary": {"passive_perc": 90},
        "redundancy": [{"phrase": "x"}] * 20,
        "dialogue_punctuation_errors": [{"text": "x"}] * 20,
        "advanced_features": {"cliches": [{"text": "x"}] * 20},
        "pro_editing": {"filter_verbs": [{"text": "x"}] * 20},
    }
    score = calculate_prose_score(report)
    assert 0 <= score <= 20


# --------------------------------------------------------------------------- #
# Dialogue ratio (#7 regression lock)
# --------------------------------------------------------------------------- #

def test_dialogue_ratio_ignores_apostrophes():
    # Pure narration full of apostrophes -> NOT dialogue
    text = "The dog wagged it's tail. The cat's bowl was empty. Don't worry."
    assert calculate_dialogue_ratio(text) == 0.0


def test_dialogue_ratio_counts_real_dialogue():
    text = '"Stop right there," she said.'
    assert calculate_dialogue_ratio(text) > 0


def test_dialogue_ratio_smart_quotes():
    text = "“Hello there,” he said. It’s a nice day."
    ratio = calculate_dialogue_ratio(text)
    # The quoted span counts; the apostrophe in "It's" does not start a quote.
    assert 0 < ratio < 60


def test_dialogue_ratio_empty():
    assert calculate_dialogue_ratio("") == 0


# --------------------------------------------------------------------------- #
# Redundancy (regex based, no NLP)
# --------------------------------------------------------------------------- #

def test_redundancy_detects_pleonasm():
    text = "He nodded his head and shrugged his shoulders."
    found = get_redundancy_analysis(text, get_sentences(text))
    phrases = {f["phrase"] for f in found}
    assert "nodded his head" in phrases
    assert "shrugged his shoulders" in phrases


def test_redundancy_none_when_clean():
    text = "He nodded and walked away."
    assert get_redundancy_analysis(text, get_sentences(text)) == []


# --------------------------------------------------------------------------- #
# Possessive its/it's (#5 regression lock) — needs a real doc
# --------------------------------------------------------------------------- #

def test_possessive_flags_its_error():
    text = "It's tail was wagging."
    doc = get_spacy_doc(text)
    starts = linter.get_sentence_starts(doc)
    errors = get_possessive_errors_analysis(text, doc, starts)
    assert len(errors) == 1
    assert errors[0]["text"] == "It's tail"
    assert errors[0]["suggestion"] == "its tail"


@pytest.mark.parametrize("text", [
    "It's gone now.",
    "It's cold outside.",
    "It's a problem.",
    "It's been raining.",
    "The dog chased its tail.",
])
def test_possessive_ignores_correct_usage(text):
    doc = get_spacy_doc(text)
    starts = linter.get_sentence_starts(doc)
    assert get_possessive_errors_analysis(text, doc, starts) == []


def test_possessive_no_doc_returns_empty():
    assert get_possessive_errors_analysis("It's tail.", None, []) == []


# --------------------------------------------------------------------------- #
# Spelling (#6 regression lock)
# --------------------------------------------------------------------------- #

def test_spelling_flags_capitalized_typo():
    # "Teh" at sentence start was missed before the case-sensitivity fix.
    text = "Teh cat ran home."
    doc = get_spacy_doc(text)
    flagged = {item["word"] for item in get_spelling_analysis(text, doc=doc)}
    assert "Teh" in flagged


def test_spelling_skips_names_and_acronyms():
    text = "Sarah met Gandalf. NASA is real."
    doc = get_spacy_doc(text)
    flagged = {item["word"] for item in get_spelling_analysis(text, doc=doc)}
    assert "Sarah" not in flagged
    assert "Gandalf" not in flagged
    assert "NASA" not in flagged


def test_spelling_known_words_suppressed():
    text = "The cyberpunk holo flickered."
    doc = get_spacy_doc(text)
    flagged = {item["word"] for item in get_spelling_analysis(text, doc=doc)}
    assert "cyberpunk" not in flagged
    assert "holo" not in flagged


def test_spelling_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(linter, "SPELLING_AVAILABLE", False)
    assert get_spelling_analysis("Teh recieve teh ball.") == []


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def test_lint_prose_report_structure():
    text = ("I raised my guns. I fired at the mouth. I felt the heat. "
            "It's tail was wagging.")
    report = lint_prose(text)

    summary = report["executive_summary"]
    assert summary["word_count"] > 0
    assert "prose_score" in summary
    assert "drone_factor" in summary
    assert isinstance(summary["total_issues"], int)

    # Repetitive "I" openings should be detected (3+ in a row).
    assert report["repetitive_structure"]
    # The possessive error should surface in the full pipeline too.
    assert any(e["text"] == "It's tail" for e in report["possessive_errors"])


def test_lint_prose_unchanged_incremental_returns_cached():
    text = "A short stable sentence."
    report = lint_prose(text)
    same = linter.incremental_lint(text, text, report)
    assert same is report
