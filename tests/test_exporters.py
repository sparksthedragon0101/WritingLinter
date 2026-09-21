import os
import pytest
from exporter import export_to_markdown, export_to_html, export_to_obsidian, export_to_text, export_to_json

@pytest.fixture
def sample_data():
    text = "This is a sample text for testing exporters. It has a few sentences."
    report = {
        "executive_summary": {
            "word_count": 13,
            "reading_time": "3s",
            "avg_sent_len": 6.5,
            "dialogue_ratio": 0.0,
            "passive_perc": 0.0,
            "prose_score": 85,
            "total_issues": 2
        },
        "fixed_text": "This is a sample text for testing exporters. It has several sentences.",
        "readability": {"score": 80},
        "tone": {"mood": "Neutral"}
    }
    report_content = "Technical analysis details..."
    return text, report, report_content

def test_markdown_export(sample_data, tmp_path):
    text, report, report_content = sample_data
    filepath = tmp_path / "test_report.md"
    export_to_markdown(text, report, report_content, str(filepath))
    
    assert os.path.exists(filepath)
    content = filepath.read_text()
    assert "Writing Linter Analysis Report" in content
    assert "Prose Score: 85/100" in content
    assert "Total Issues Found | 2" in content
    assert "AI Refined Text" in content

def test_html_export(sample_data, tmp_path):
    text, report, report_content = sample_data
    filepath = tmp_path / "test_report.html"
    # Note: this might fail if templates dir is not found relative to exporter.py
    # But in tests, we should ensure the environment is set up.
    export_to_html(text, report, report_content, str(filepath))
    
    assert os.path.exists(filepath)
    content = filepath.read_text()
    # Check for keywords in our new premium template
    assert "Writing Linter Analysis" in content
    assert "Overall Prose Score" in content
    assert "85" in content

def test_obsidian_export(sample_data, tmp_path):
    text, report, report_content = sample_data
    filepath = tmp_path / "test_report_obsidian.md"
    export_to_obsidian(text, report, report_content, str(filepath))
    
    assert os.path.exists(filepath)
    content = filepath.read_text()
    assert "---" in content
    assert "tags: [writing-linter, report]" in content
    assert "[!stats] Analysis Results" in content
    assert "Prose Score:** 85/100" in content

def test_text_export(sample_data, tmp_path):
    text, report, report_content = sample_data
    filepath = tmp_path / "test_report.txt"
    export_to_text(text, report, report_content, str(filepath))
    
    assert os.path.exists(filepath)
    content = filepath.read_text()
    assert "Executive Summary" in content
    assert "Word Count: 13" in content

def test_json_export(sample_data, tmp_path):
    text, report, report_content = sample_data
    filepath = tmp_path / "test_report.json"
    export_to_json(text, report, str(filepath))
    
    assert os.path.exists(filepath)
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    assert data["version"] == "1.0"
    assert data["metadata"]["word_count"] == 13
    assert data["text_content"]["original"] == text
