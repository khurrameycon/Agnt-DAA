"""
Model manager for sagax1
Handles loading, caching, and managing models
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional
import threading
from pathlib import Path
import shutil

import huggingface_hub
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from app.core.config_manager import ConfigManager

class ModelManager:
    """Manages AI models for the application"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the model manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Get default cache directory
        self.cache_dir = os.path.expanduser(
            config_manager.get("models.cache_dir", "~/.cache/sagax1/models")
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dictionary to store model metadata
        self.model_metadata = {}
        self.load_cached_model_metadata()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # HF API client
        self.api = HfApi(token=config_manager.get_hf_api_key())
    
    def load_cached_model_metadata(self) -> None:
        """Load cached model metadata from disk"""
        metadata_path = os.path.join(self.cache_dir, "model_metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                self.logger.info(f"Loaded metadata for {len(self.model_metadata)} models")
            except Exception as e:
                self.logger.error(f"Error loading model metadata: {str(e)}")
                self.model_metadata = {}
    
    def save_model_metadata(self) -> None:
        """Save model metadata to disk"""
        metadata_path = os.path.join(self.cache_dir, "model_metadata.json")
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving model metadata: {str(e)}")
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a model
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        # Check if we already have metadata
        if model_id in self.model_metadata:
            return self.model_metadata[model_id]
        
        # Otherwise fetch from API
        try:
            model_info = self.api.model_info(model_id)
            model_data = {
                "id": model_id,
                "name": model_id.split("/")[-1],  # Use last part of ID as name
                "downloads": model_info.downloads,
                "likes": model_info.likes,
                "tags": model_info.tags,
                "pipeline_tag": model_info.pipeline_tag,
                "size_in_bytes": model_info.size_in_bytes,
                "is_gated": "gated" in model_info.tags,
                "last_modified": model_info.last_modified.isoformat() if model_info.last_modified else None,
                "config": {}
            }
            
            # Add to metadata cache
            with self.lock:
                self.model_metadata[model_id] = model_data
                self.save_model_metadata()
            
            return model_data
        except (RepositoryNotFoundError, RevisionNotFoundError):
            self.logger.warning(f"Model '{model_id}' not found")
            return None
        except Exception as e:
            self.logger.error(f"Error getting model info: {str(e)}")
            return None
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is already downloaded
        
        Args:
            model_id: Model ID
            
        Returns:
            True if cached, False otherwise
        """
        model_path = os.path.join(self.cache_dir, model_id.replace("/", "_"))
        return os.path.exists(model_path)
    
    def download_model(self, model_id: str, revision: Optional[str] = None, force: bool = False) -> bool:
        """Download a model
        
        Args:
            model_id: Model ID
            revision: Optional revision
            force: Force re-download even if cached
            
        Returns:
            True if successful, False otherwise
        """
        if self.is_model_cached(model_id) and not force:
            self.logger.info(f"Model {model_id} already downloaded")
            return True
        
        try:
            model_path = os.path.join(self.cache_dir, model_id.replace("/", "_"))
            
            # Download the model
            self.logger.info(f"Downloading model {model_id}...")
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=model_path,
                token=self.config_manager.get_hf_api_key()
            )
            
            # Update metadata
            model_info = self.get_model_info(model_id)
            
            if model_info:
                model_info["is_cached"] = True
                self.save_model_metadata()
            
            self.logger.info(f"Model {model_id} downloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading model {model_id}: {str(e)}")
            return False
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a downloaded model
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful, False otherwise
        """
        model_path = os.path.join(self.cache_dir, model_id.replace("/", "_"))
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Model {model_id} not found in cache")
            return False
        
        try:
            # Remove the model directory
            shutil.rmtree(model_path)
            
            # Update metadata
            if model_id in self.model_metadata:
                self.model_metadata[model_id]["is_cached"] = False
                self.save_model_metadata()
            
            self.logger.info(f"Model {model_id} removed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing model {model_id}: {str(e)}")
            return False
    
    def get_cached_models(self) -> List[str]:
        """Get the list of cached models
        
        Returns:
            List of model IDs
        """
        models = []
        
        for entry in os.listdir(self.cache_dir):
            if entry != "model_metadata.json" and os.path.isdir(os.path.join(self.cache_dir, entry)):
                # Convert back from filesystem-safe name to model ID
                model_id = entry.replace("_", "/", 1)
                models.append(model_id)
        
        return models
    
    def search_models(self, query: str, filter_tags: Optional[List[str]] = None, 
                     limit: int = 20) -> List[Dict[str, Any]]:
        """Search for models on Hugging Face
        
        Args:
            query: Search query
            filter_tags: List of tags to filter by
            limit: Maximum number of results
            
        Returns:
            List of model information
        """
        try:
            # Build filter string
            filter_str = None
            if filter_tags:
                filter_str = "+".join(filter_tags)
            
            # Search models
            models = self.api.list_models(
                search=query,
                filter=filter_str,
                limit=limit,
                sort="downloads",
                direction=-1
            )
            
            # Convert to simple dictionaries
            results = []
            for model in models:
                result = {
                    "id": model.id,
                    "name": model.id.split("/")[-1],  # Use last part of ID as name if it doesn't exist
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "pipeline_tag": model.pipeline_tag,
                    "is_cached": self.is_model_cached(model.id)
                }
                results.append(result)
                
                # Update metadata cache
                if model.id not in self.model_metadata:
                    self.model_metadata[model.id] = result
            
            # Save updated metadata
            self.save_model_metadata()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching models: {str(e)}")
            return []
    
    def get_model_tags(self) -> Dict[str, int]:
        """Get popular model tags and their counts
        
        Returns:
            Dictionary of tags and counts
        """
        try:
            # We'll use the metadata we have
            tags = {}
            
            for model_id, info in self.model_metadata.items():
                for tag in info.get("tags", []):
                    if tag in tags:
                        tags[tag] += 1
                    else:
                        tags[tag] = 1
            
            # Sort by count and return top tags
            return dict(sorted(tags.items(), key=lambda x: x[1], reverse=True)[:50])
            
        except Exception as e:
            self.logger.error(f"Error getting model tags: {str(e)}")
            return {}