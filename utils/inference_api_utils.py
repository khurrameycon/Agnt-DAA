"""
Utilities for fetching and managing Inference API models
This can be saved as app/utils/inference_api_utils.py
"""

import requests
import logging
import json
import os
import time
from typing import List, Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)

# Cache to avoid excessive API calls
_inference_models_cache = None
_cache_timestamp = 0
_cache_lock = Lock()
_cache_lifetime = 3600  # Cache lifetime in seconds (1 hour)

def get_inference_models(force_refresh: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
    """Get models that support the Hugging Face Inference API
    
    Args:
        force_refresh: Whether to force a refresh of the cache
        limit: Maximum number of models to return
        
    Returns:
        List of model information dictionaries
    """
    global _inference_models_cache, _cache_timestamp
    
    # Check if we can use the cache
    current_time = time.time()
    with _cache_lock:
        if not force_refresh and _inference_models_cache is not None:
            if current_time - _cache_timestamp < _cache_lifetime:
                logger.info(f"Using cached inference models (cache age: {int(current_time - _cache_timestamp)}s)")
                return _inference_models_cache
    
    logger.info(f"Fetching inference API models from Hugging Face (limit: {limit})")
    
    try:
        # Make the API request
        url = f"https://huggingface.co/api/models?limit={limit}&inference_endpoints=true"
        
        # Add API token if available (for higher rate limits)
        headers = {}
        api_token = os.environ.get("HF_API_KEY")
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Successfully fetched {len(models)} inference API models")
            
            # Update the cache
            with _cache_lock:
                _inference_models_cache = models
                _cache_timestamp = current_time
            
            return models
        else:
            logger.error(f"Error fetching inference models: {response.status_code} - {response.text}")
            
            # Return empty list or cached data if available
            with _cache_lock:
                if _inference_models_cache is not None:
                    logger.info("Using stale cache due to API error")
                    return _inference_models_cache
            
            return []
    
    except Exception as e:
        logger.error(f"Exception when fetching inference models: {str(e)}")
        
        # Return empty list or cached data if available
        with _cache_lock:
            if _inference_models_cache is not None:
                logger.info("Using stale cache due to exception")
                return _inference_models_cache
        
        return []

def search_inference_models(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search for inference API models
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of model information dictionaries
    """
    logger.info(f"Searching inference API models for: {query} (limit: {limit})")
    
    try:
        # Make the API request
        url = f"https://huggingface.co/api/models?search={query}&limit={limit}&inference_endpoints=true"
        
        # Add API token if available (for higher rate limits)
        headers = {}
        api_token = os.environ.get("HF_API_KEY")
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Search returned {len(models)} inference API models")
            return models
        else:
            logger.error(f"Error searching inference models: {response.status_code} - {response.text}")
            return []
    
    except Exception as e:
        logger.error(f"Exception when searching inference models: {str(e)}")
        return []

def is_model_available_for_inference(model_id: str) -> bool:
    """Check if a specific model is available for inference
    
    Args:
        model_id: Model ID to check
        
    Returns:
        True if the model is available for inference, False otherwise
    """
    logger.info(f"Checking if model is available for inference: {model_id}")
    
    try:
        # Make the API request
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        # Add API token if available
        headers = {}
        api_token = os.environ.get("HF_API_KEY")
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        
        response = requests.get(url, headers=headers)
        
        # Status code 200 means model exists and is available
        if response.status_code == 200:
            logger.info(f"Model {model_id} is available for inference")
            return True
        
        # Status code 404 means model doesn't exist
        elif response.status_code == 404:
            logger.info(f"Model {model_id} not found for inference")
            return False
        
        # Status code 503 often means model is loading
        elif response.status_code == 503:
            logger.info(f"Model {model_id} exists but is currently loading")
            return True  # It exists, just needs time to load
        
        # Other status codes
        else:
            logger.warning(f"Unexpected status code when checking model {model_id}: {response.status_code}")
            
            # Check response text for clues
            if "not found" in response.text.lower():
                return False
            
            # Assume it might be available but with some issue
            return True
    
    except Exception as e:
        logger.error(f"Exception when checking model {model_id} for inference: {str(e)}")
        return False