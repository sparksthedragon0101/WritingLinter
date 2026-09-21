import pytest
import spacy
from text_refiner import get_spacy_doc
from linter import extract_and_blank_suppressions, filter_suppressed_issues, fix_prose

def test_extract_and_blank_suppressions():
    text = """This is normal text.
<!-- lint-disable-next-line passive_voice -->
This text has an issue.
This is back to normal.
<!-- lint-disable pro_editing -->
More text.
<!-- lint-enable pro_editing -->"""
    blanked, suppressions = extract_and_blank_suppressions(text)
    assert len(suppressions) > 0

    assert any(s["issue"] == "passive_voice" for s in suppressions)
    assert any(s["issue"] == "pro_editing" for s in suppressions)

def test_filter_suppressed_issues():
    text = "abc <!-- lint-disable redundancy --> bad text <!-- lint-enable redundancy --> xyz"
    blanked, suppressions = extract_and_blank_suppressions(text)
    
    bad_text_start = text.find("bad")
    bad_text_end = bad_text_start + 8
    
    report = {
        "redundancy": [{"start": bad_text_start, "end": bad_text_end, "text": "bad text"}]
    }
    filtered, count = filter_suppressed_issues(report, suppressions)
    assert count == 1
    assert len(filtered["redundancy"]) == 0

def test_fix_prose():
    text = '"Stop." he said.'
    doc = get_spacy_doc(text)
    fixed = fix_prose(text, doc)
    assert fixed == '"Stop," he said.'
