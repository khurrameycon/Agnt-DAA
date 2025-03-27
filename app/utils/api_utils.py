"""
API utilities for SagaX1
Handles interactions with external APIs
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
import requests
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

class HuggingFaceAPI:
    """Handles interactions with the Hugging Face API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Hugging Face API
        
        Args:
            api_key: Hugging Face API key
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.environ.get("HF_API_KEY") or HfFolder.get_token()
        self.api = HfApi(token=self.api_key)
    
    def validate_api_key(self) -> bool:
        """Validate the API key
        
        Returns:
            True if the API key is valid, False otherwise
        """
        if not self.api_key:
            self.logger.warning("No API key provided")
            return False
        
        try:
            # Make a simple API call to test the key
            self.api.whoami()
            return True
        except Exception as e:
            self.logger.error(f"Invalid API key: {str(e)}")
            return False
    
    def search_models(self, query: str, filter: Optional[str] = None, 
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Search for models on Hugging Face
        
        Args:
            query: Search query
            filter: Filter for model search
            limit: Maximum number of results
            
        Returns:
            List of model information
        """
        try:
            models = self.api.list_models(
                search=query,
                filter=filter,
                limit=limit
            )
            
            return [model.to_dict() for model in models]
        except Exception as e:
            self.logger.error(f"Error searching models: {str(e)}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a model
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        try:
            model_info = self.api.model_info(model_id)
            return model_info.to_dict()
        except (RepositoryNotFoundError, RevisionNotFoundError):
            self.logger.warning(f"Model '{model_id}' not found")
            return None
        except Exception as e:
            self.logger.error(f"Error getting model info: {str(e)}")
            return None
    
    def get_model_tags(self, model_id: str) -> List[str]:
        """Get tags for a model
        
        Args:
            model_id: Model ID
            
        Returns:
            List of tags
        """
        model_info = self.get_model_info(model_id)
        
        if model_info:
            return model_info.get("tags", [])
        
        return []
    
    def is_model_downloadable(self, model_id: str) -> bool:
        """Check if a model is downloadable
        
        Args:
            model_id: Model ID
            
        Returns:
            True if downloadable, False otherwise
        """
        model_info = self.get_model_info(model_id)
        
        if not model_info:
            return False
            
        # Check if model requires authentication
        if model_info.get("private", False) and not self.api_key:
            return False
            
        # Check if model is gated
        if "gated" in model_info.get("tags", []) and not self.api_key:
            return False
            
        return True

    def get_compatible_spaces(self, task: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find compatible Hugging Face Spaces for a specific task
        
        Args:
            task: Task to find Spaces for (e.g., "image-generation", "text-to-speech")
            limit: Maximum number of results
            
        Returns:
            List of Space information
        """
        try:
            spaces = self.api.list_spaces(
                search=task,
                limit=limit,
                sort="likes"
            )
            
            return [space.to_dict() for space in spaces]
        except Exception as e:
            self.logger.error(f"Error getting compatible spaces: {str(e)}")
            return []