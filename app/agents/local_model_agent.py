"""
Local Model Agent for sagax1
Runs local Hugging Face models for text generation and chat interactions
Enhanced with better Inference API support
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, Callable

from app.agents.base_agent import BaseAgent
from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
from huggingface_hub import snapshot_download

class LocalModelAgent(BaseAgent):
    """Agent for running local models from Hugging Face for chat and text generation"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the local model agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                use_api: Whether to use the Hugging Face Inference API (remote execution)
                use_local_execution: Whether to use local execution (download model)
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
        self.device = config.get("device", "auto")
        self.max_new_tokens = config.get("max_tokens", 2048)  # Changed to max_new_tokens
        self.temperature = config.get("temperature", 0.1)
        self.authorized_imports = config.get("authorized_imports", [])
        
        # Get execution mode - prioritize explicit flags
        self.use_api = config.get("use_api", False)
        self.use_local_execution = config.get("use_local_execution", not self.use_api)
        
        # If both flags are somehow set (shouldn't happen), prioritize API mode
        if self.use_api and self.use_local_execution:
            self.logger.warning("Both use_api and use_local_execution are set to True. Prioritizing API mode.")
            self.use_local_execution = False
        
        self.model = None
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the model based on config - local or API"""
        if self.is_initialized:
            return
        
        self.logger.info(f"Initializing model {self.model_id} with mode: {'API' if self.use_api else 'Local'}")
        
        try:
            if self.use_api:
                # Use API mode - don't download the model at all
                self._initialize_api_model()
            else:
                # Local execution mode - download model and use locally
                self._ensure_model_downloaded()
                self._initialize_local_model()
                
            self.is_initialized = True
            self.logger.info(f"Model {self.model_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Only try fallbacks for local execution - don't automatically download if API fails
            if not self.use_api:
                raise # self._initialize_with_fallbacks()
            else:
                # For API mode, just raise the error without downloading
                raise

    def _initialize_api_model(self):
        """Initialize the model by wrapping a direct HTTP call to the HF Inference API."""
        from app.core.config_manager import ConfigManager
        import requests

        # 1. Retrieve API key
        api_key = ConfigManager().get_hf_api_key()
        self.logger.info(f"Loaded HF API key: {api_key!r}")
        if not api_key:
            self.logger.error("No API key found. Cannot use Inference API.")
            raise ValueError("HuggingFace API key is required for Inference API mode")
        
        # 2. Prepare headers once
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # 3. Build the base URL for your model
        base_url = f"https://router.huggingface.co/hf-inference/models/{self.model_id}/v1/chat/completions"

        # 4. The wrapper function
        def generate_text(messages):
            try:
                # --- extract prompt from messages (your existing logic) ---
                prompt = ""
                if isinstance(messages, list) and messages:
                    last = messages[-1]
                    if isinstance(last, dict) and "content" in last:
                        content = last["content"]
                        if isinstance(content, list):
                            prompt = " ".join(
                                item.get("text", "")
                                for item in content
                                if item.get("type") == "text"
                            )
                        else:
                            prompt = content
                    else:
                        prompt = str(last)
                # ----------------------------------------------------------

                # 5. Build payload exactly as in your test
                payload = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_new_tokens,
                    "temperature": float(self.temperature),
                }

                # 6. POST to the inference endpoint
                resp = requests.post(base_url, headers=headers, json=payload)

                # 7. Error handling
                if resp.status_code != 200:
                    self.logger.error(f"Inference API HTTP {resp.status_code}: {resp.text}")
                    return f"Error calling Inference API: {resp.status_code} {resp.text}"

                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                self.logger.error(f"Inference API exception: {e}")
                return f"Error calling Inference API: {e}"

        # 8. Bind it and log
        self.model = generate_text
        self.logger.info(f"Initialized {self.model_id} via direct HTTP inference")



    def _initialize_local_model(self):
        """Initialize the model locally"""
        from smolagents import TransformersModel
        
        self.model = TransformersModel(
            model_id=self.model_id,
            device_map=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            trust_remote_code=True,
            do_sample=True  # Add this to fix the temperature warning
        )
        
        self.logger.info(f"Initialized {self.model_id} for local execution")
    
    def _initialize_with_fallbacks(self):
        """Try alternative model implementations if TransformersModel fails"""
        try:
            # Try HfApiModel
            try:
                self.logger.info("Trying HfApiModel...")
                self.model = HfApiModel(
                    model_id=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize with HfApiModel: {str(e)}")
                
                # Try OpenAIServerModel
                try:
                    self.logger.info("Trying OpenAIServerModel...")
                    self.model = OpenAIServerModel(
                        model_id=self.model_id,
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to initialize with OpenAIServerModel: {str(e)}")
                    
                    # Try LiteLLMModel as last resort
                    self.logger.info("Trying LiteLLMModel...")
                    self.model = LiteLLMModel(
                        model_id=self.model_id,
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens
                    )
            
            self.is_initialized = True
            self.logger.info(f"Model {self.model_id} initialized successfully with fallback")
            
        except Exception as e:
            self.logger.error(f"All fallback initialization attempts failed: {str(e)}")
            raise
    
    def _ensure_model_downloaded(self) -> None:
        """Download the model if needed"""
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            
            # Try to download model card first as a test
            hf_hub_download(
                repo_id=self.model_id,
                filename="config.json",
                token=os.environ.get("HF_API_KEY")
            )
            
            # If successful, model is available
            self.logger.info(f"Model {self.model_id} is available")
            
        except Exception as e:
            self.logger.error(f"Error checking model availability: {str(e)}")
            raise
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the model with the given input
        
        Args:
            input_text: Input text for the model
            callback: Optional callback for streaming responses
            
        Returns:
            Model output text
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Format the input in the format expected by the model
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": input_text
                        }
                    ]
                }
            ]
            
            # If using API mode and callback is provided, show "Processing with API..." message
            if self.use_api and callback:
                callback("Processing with Hugging Face Inference API...")
            
            # Call the model with the correctly formatted messages
            response = self.model(messages)
            
            # Convert the response to a string based on its type
            if hasattr(response, 'content'):
                # If it's a ChatMessage object with a content attribute
                result_text = response.content
            elif hasattr(response, 'text'):
                # If it has a text attribute
                result_text = response.text
            elif hasattr(response, '__str__'):
                # Fall back to string representation
                result_text = str(response)
            else:
                # Last resort fallback
                result_text = "Response received but could not be converted to text"
            
            # Add to history
            self.add_to_history(input_text, result_text)
            
            return result_text
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error: {error_msg}"
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        return ["text_generation", "conversational_chat"]