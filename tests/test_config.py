import os
import yaml
import pytest
from unittest.mock import patch, mock_open
from config import load_config, save_accepted_pattern

def test_load_config_default():
    with patch("os.path.exists", return_value=False):
        cfg = load_config()
        assert cfg["ignore_rules"] == []

def test_load_config_with_local(tmp_path):
    local_cfg_content = {
        "model": "en_core_web_sm",
        "ignore_rules": ["PASSIVE_VOICE"]
    }
    
    # Create a local config in a temporary directory
    local_config_file = tmp_path / ".writing_linter.yaml"
    with open(local_config_file, "w") as f:
        yaml.dump(local_cfg_content, f)
    
    with patch("os.path.exists") as mock_exists:
        # Mock global not existing, local existing
        def exists_side_effect(path):
            if ".writing_linter.yaml" in str(path) and "home" not in str(path):
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(local_cfg_content))):
            cfg = load_config(workspace_dir=str(tmp_path))
            assert cfg["model"] == "en_core_web_sm"
            assert "PASSIVE_VOICE" in cfg["ignore_rules"]

def test_save_accepted_pattern(tmp_path):
    local_config_file = tmp_path / ".writing_linter.yaml"
    
    # Ensure it starts empty or doesn't exist
    if os.path.exists(local_config_file):
        os.remove(local_config_file)
        
    pattern = r"test_pattern_\d+"
    success = save_accepted_pattern(pattern, workspace_dir=str(tmp_path))
    
    assert success is True
    assert os.path.exists(local_config_file)
    
    with open(local_config_file, "r") as f:
        saved_cfg = yaml.safe_load(f)
    assert pattern in saved_cfg["accepted_patterns"]
