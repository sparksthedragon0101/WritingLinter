"""
Golden-corpus regression test.

tests/corpus/golden.txt contains deliberately planted issues, one group per
paragraph. This suite pins the analysis output against those known plants so
heuristic tuning can't silently change what the engine detects.

Pinning policy:
- exact assertions for regex/deterministic checks (cliches, adverbial tags,
  possessive errors, spelling, run-ons, clipped clusters, I-clusters)
- bounded assertions for spaCy-parse-dependent checks (passive voice,
  filter verbs), so a model upgrade doesn't spuriously fail
- zero assertions for checks the corpus should NOT trigger, guarding
  against false-positive regressions
"""
import os

import pytest

import linter

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus", "golden.txt")


@pytest.fixture(scope="module")
def corpus_text():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def report(corpus_text):
    return linter.lint_prose(corpus_text)


# --------------------------------------------------------------------------- #
# Planted issues must be detected
# --------------------------------------------------------------------------- #

def test_passive_voice_detected(report):
    pv = report["passive_voice"]
    # Planted: "was written by John", "was buried by pirates", "was met with"
    assert 3 <= pv["passive_count"] <= 5
    assert pv["passive_percentage"] > 0
    example_texts = " ".join(e.get("text", "") for e in pv.get("examples", []))
    assert "written" in example_texts
    assert "buried" in example_texts


def test_repetitive_i_cluster_detected(report):
    # Planted: exactly one run of 3 consecutive sentences starting with "I ".
    clusters = report["repetitive_structure"]
    assert len(clusters) == 1
    assert clusters[0]["count"] == 3


def test_run_on_detected(report):
    # Planted: exactly one 40-word detective sentence.
    run_ons = report["run_ons"]
    assert len(run_ons) == 1
    assert run_ons[0]["word_count"] == 40
    assert run_ons[0]["text"].startswith("The detective walked slowly")


def test_clipped_cluster_detected(report):
    # Planted: one run of 3 consecutive short sentences (He ran. / She
    # followed. / The door slammed.)
    clusters = report["clipped_sentences"]
    assert len(clusters) == 1
    assert [item["text"] for item in clusters[0]] == [
        "He ran.", "She followed.", "The door slammed."
    ]


def test_cliches_detected(report):
    # Planted: exactly these two, in this order.
    cliches = report["advanced_features"]["cliches"]
    assert [c["text"] for c in cliches] == ["dead of night", "cold as ice"]


def test_adverb_heavy_sentence_detected(report):
    # Planted: one sentence with two -ly adverbs ("carefully and quietly").
    heavy = report["advanced_features"]["adverb_heavy_sentences"]
    assert len(heavy) == 1
    assert set(heavy[0]["adverbs"]) == {"carefully", "quietly"}


def test_filter_verbs_detected(report):
    # Planted: looked, heard, saw, felt, watched.
    filters = report["pro_editing"]["filter_verbs"]
    assert 4 <= len(filters) <= 6


def test_show_dont_tell_detected(report):
    issues = report["show_dont_tell"]
    phrases = {i["phrase"] for i in issues if i["type"] == "filter"}
    assert {"heard", "saw", "felt", "watched"} <= phrases
    # Planted emotion-telling: "She was angry."
    assert any(i["type"] == "emotion" for i in issues)


def test_adverbial_dialogue_tags_detected(report):
    # Planted: exactly these two.
    tags = report["adverbial_tags"]
    assert [t["text"] for t in tags] == ["said angrily", "whispered softly"]


def test_possessive_error_detected(report):
    # Planted: "It's tail was wagging."
    errors = report["possessive_errors"]
    assert len(errors) == 1
    assert errors[0]["text"] == "It's tail"


def test_spelling_error_detected(report):
    # Planted: "Teh".
    misspelled = {s["word"] for s in report["spelling"]}
    assert misspelled == {"Teh"}


def test_dangling_participle_detected(report):
    # Planted: "Running down the street, the bag fell from her hands."
    assert len(report["dangling_participles"]) == 1


def test_comma_splices_detected(report):
    # Planted: exactly two ("The lamp flickered, the room went dark." and
    # "He reached for the switch, nothing happened."). The subordinate clause
    # in the same paragraph ("After the power died, she lit a candle") must
    # not be counted.
    splices = report["comma_splices"]
    assert len(splices) == 2
    assert "flickered" in splices[0]["text"]
    assert "nothing happened" in splices[1]["text"]


def test_word_echo_detected(report):
    # Planted: "miles and miles" inside the run-on sentence.
    echoed_lemmas = {lemma for lemma, _ in report["advanced_features"]["word_echoes"]}
    assert "mile" in echoed_lemmas


# --------------------------------------------------------------------------- #
# Checks the corpus must NOT trigger (false-positive guards)
# --------------------------------------------------------------------------- #

def test_no_false_positives(report):
    assert report["sentence_splices"] == []
    assert report["dialogue_punctuation_errors"] == []
    assert report["tense_consistency"] == []
    assert report["redundancy"] == []


# --------------------------------------------------------------------------- #
# Report integrity
# --------------------------------------------------------------------------- #

def test_offsets_match_source_text(report, corpus_text):
    """
    Every start/end offset must slice back to the reported text. Offsets
    cover the raw sentence span (which may include trailing whitespace);
    reported text is the stripped form.
    """
    run_on = report["run_ons"][0]
    assert corpus_text[run_on["start"]:run_on["end"]].strip() == run_on["text"]

    for item in report["clipped_sentences"][0]:
        assert corpus_text[item["start"]:item["end"]].strip() == item["text"]

    for err in report["possessive_errors"]:
        assert corpus_text[err["start"]:err["end"]] == err["text"]


def test_summary_metrics(report):
    summ = report["executive_summary"]
    assert summ["word_count"] == 209
    # The corpus is deliberately flawed; score must reflect that without
    # bottoming out. Observed baseline: 72.
    assert 55 <= summ["prose_score"] <= 85
    assert summ["total_issues"] >= 12


# --------------------------------------------------------------------------- #
# Auto-fix golden behavior
# --------------------------------------------------------------------------- #

def test_fix_prose_corrects_possessive(corpus_text):
    fixed = linter.fix_prose(corpus_text)
    assert "Its tail was wagging." in fixed
    assert "It's tail" not in fixed
    # The fixer must not mangle surrounding text.
    assert "dog barked at the moon." in fixed
