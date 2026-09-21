import os
import yaml

DEFAULT_CONFIG = {
    "ignore_rules": [],
    "accepted_patterns": [],
    "model": "en_core_web_md"
}

def load_config(workspace_dir=None):
    """
    Loads configuration from ~/.writing_linter.yaml and a local .writing_linter.yaml
    Local config overrides global config.
    """
    config = DEFAULT_CONFIG.copy()
    
    global_config_path = os.path.expanduser("~/.writing_linter.yaml")
    if os.path.exists(global_config_path):
        try:
            with open(global_config_path, "r", encoding="utf-8") as f:
                global_config = yaml.safe_load(f) or {}
                _merge_config(config, global_config)
        except Exception as e:
            print(f"Warning: Failed to load global config: {e}")

    local_config_path = ".writing_linter.yaml"
    if workspace_dir:
        local_config_path = os.path.join(workspace_dir, ".writing_linter.yaml")
        
    if os.path.exists(local_config_path):
        try:
            with open(local_config_path, "r", encoding="utf-8") as f:
                local_config = yaml.safe_load(f) or {}
                _merge_config(config, local_config)
        except Exception as e:
            print(f"Warning: Failed to load local config: {e}")

    return config

def _merge_config(base, new_config):
    if "ignore_rules" in new_config:
        base["ignore_rules"].extend(new_config["ignore_rules"])
        base["ignore_rules"] = list(set(base["ignore_rules"]))
        
    if "accepted_patterns" in new_config:
        base["accepted_patterns"].extend(new_config["accepted_patterns"])
        base["accepted_patterns"] = list(set(base["accepted_patterns"]))
        
    if "model" in new_config:
        base["model"] = new_config["model"]

def save_accepted_pattern(pattern, workspace_dir=None):
    """
    Saves a new accepted pattern to the local .writing_linter.yaml config.
    """
    local_config_path = ".writing_linter.yaml"
    if workspace_dir:
        local_config_path = os.path.join(workspace_dir, ".writing_linter.yaml")
        
    config = {}
    if os.path.exists(local_config_path):
        try:
            with open(local_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load local config for saving: {e}")
            
    if "accepted_patterns" not in config:
        config["accepted_patterns"] = []
        
    if pattern not in config["accepted_patterns"]:
        config["accepted_patterns"].append(pattern)
        try:
            with open(local_config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Successfully saved pattern '{pattern}' to {local_config_path}")
            return True
        except Exception as e:
            print(f"Error: Failed to save to local config: {e}")
            
    return False
