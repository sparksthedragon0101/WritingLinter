import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
from writing_linter_cli import main

def test_cli_help():
    with patch("sys.argv", ["writing_linter_cli.py", "--help"]):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            try:
                main()
            except SystemExit:
                pass
            assert "Writing Linter CLI" in fake_out.getvalue()

def test_cli_lint_json():
    sample_text = "The quick brown fox jumps over the lazy dog."
    with patch("sys.argv", ["writing_linter_cli.py", "lint", "--json", "--compact"]):
        with patch("sys.stdin", StringIO(sample_text)):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                # Mock linter.lint_prose to avoid heavy NLP during CLI test
                mock_report = {
                    "executive_summary": {"word_count": 9, "prose_score": 90},
                    "passive_voice": []
                }
                with patch("linter.lint_prose", return_value=mock_report):
                    main()
                    output = fake_out.getvalue()
                    assert "es" in output  # Compact key for executive_summary
                    data = json.loads(output)
                    assert data["es"]["word_count"] == 9

def test_cli_tts_check():
    sample_text = "Big black bug bit a big black bear."
    with patch("sys.argv", ["writing_linter_cli.py", "tts-check", "--read-aloud-preview"]):
        with patch("sys.stdin", StringIO(sample_text)):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                mock_tts_report = {"phoneme_collisions": [{"text": "big black bug"}]}
                with patch("tts_analyzer.run_tts_analysis", return_value=mock_tts_report):
                    with patch("tts_analyzer.generate_read_aloud_preview", return_value="[!] Preview text"):
                        main()
                        assert "[!] Preview text" in fake_out.getvalue()

def test_cli_fix():
    sample_text = "It's a trap."
    with patch("sys.argv", ["writing_linter_cli.py", "fix"]):
        with patch("sys.stdin", StringIO(sample_text)):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                with patch("linter.fix_prose", return_value="Fixed text"):
                    main()
                    assert "Fixed text" in fake_out.getvalue()
