"""
Configuration manager for SagaX1
Handles loading, saving, and managing application configuration
"""

import os
import json
import logging
from typing import Dict, Any, Optional

DEFAULT_CONFIG = {
    "api_keys": {
        "huggingface": ""
    },
    "models": {
        "default_model": "meta-llama/Llama-3.2-3B-Instruct",
        "cache_dir": "~/.cache/sagax1/models"
    },
    "ui": {
        "theme": "light",
        "font_size": 12
    },
    "agents": {
        "default_agent": "text_completion",
        "max_history": 100
    },
    "execution": {
        "python_executor": "local",  # Options: local, docker, e2b
        "max_execution_time": 30,    # seconds
        "authorized_imports": [
            "numpy", "pandas", "matplotlib", "PIL", "requests", 
            "bs4", "datetime", "math", "re", "os", "csv", "json"
        ]
    },
    "web": {
        "browser_user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "browser_width": 1280,
        "browser_height": 800
    }
}

class ConfigManager:
    """Manages the application configuration"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default if it doesn't exist"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}")
                print("Using default configuration")
                return DEFAULT_CONFIG.copy()
        else:
            # Create default config file
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            self._save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key
        
        Args:
            key: Key path using dot notation (e.g., 'agents.default_agent')
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, save: bool = True) -> None:
        """Set a configuration value by key
        
        Args:
            key: Key path using dot notation (e.g., 'agents.default_agent')
            value: Value to set
            save: Whether to save the configuration to file
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the right level
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        if save:
            self._save_config(self.config)
    
    def get_hf_api_key(self) -> Optional[str]:
        """Get the Hugging Face API key
        
        Returns:
            The API key or None if not set
        """
        # First check environment variable
        api_key = os.environ.get("HF_API_KEY")
        
        # If not in environment, check config
        if not api_key:
            api_key = self.get("api_keys.huggingface")
        
        return api_key if api_key else None
    
    def set_hf_api_key(self, api_key: str) -> None:
        """Set the Hugging Face API key
        
        Args:
            api_key: The API key to set
        """
        self.set("api_keys.huggingface", api_key)
    
    def save(self) -> None:
        """Save the current configuration to file"""
        self._save_config(self.config)