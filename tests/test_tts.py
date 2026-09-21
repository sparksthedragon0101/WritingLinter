import pytest
from tts_analyzer import check_phoneme_collision, check_tongue_twisters, check_breath_pacing

def test_phoneme_collisions():
    text = "The big game was at the bus station."
    collisions = check_phoneme_collision(text)
    # the function should identify heavy s/sh sounds
    assert len(collisions) > 0
    assert "big game" in collisions[0]["text"].lower() or "bus station" in collisions[0]["text"].lower()

def test_tongue_twisters():
    text = "Peter Piper picked a peck of pickled peppers."
    twisters = check_tongue_twisters(text)
    assert len(twisters) > 0
    assert "peter piper" in twisters[0]["text"].lower() or "pick" in twisters[0]["text"].lower()

def test_breath_pacing():
    text = "This is a very long sentence that just keeps going on and on without any punctuation whatsoever to allow the narrator to take a breath and it might cause them to run out of air before they reach the end."
    breath_issues = check_breath_pacing(text)
    assert len(breath_issues) > 0
