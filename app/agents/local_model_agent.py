"""
Local Model Agent for sagax1
Runs local Hugging Face models for text generation and chat interactions
"""

import os
import logging
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
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
        self.device = config.get("device", "auto")
        self.max_new_tokens = config.get("max_tokens", 2048)  # Changed to max_new_tokens
        self.temperature = config.get("temperature", 0.1)
        self.authorized_imports = config.get("authorized_imports", [])
        
        self.model = None
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the model based on config - local or API"""
        if self.is_initialized:
            return
            
        # Check execution mode
        use_api = self.config.get("use_api", False)
        use_local_execution = self.config.get("use_local_execution", True)
        
        self.logger.info(f"Initializing model {self.model_id} with mode: " + 
                        ("API" if use_api else "Local"))
        
        try:
            if use_api:
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
            if not use_api:
                self._initialize_with_fallbacks()
            else:
                # For API mode, just raise the error without downloading
                raise

    def _initialize_api_model(self):
        """Initialize the model using a direct REST API request to Hugging Face Inference"""
        # Get API key from environment or config
        api_key = os.environ.get("HF_API_KEY")
        if not api_key:
            # Try to get from config manager if available
            try:
                if hasattr(self, 'config_manager') and self.config_manager is not None:
                    api_key = self.config_manager.get_hf_api_key()
            except:
                pass
        
        if not api_key:
            self.logger.warning("No API key found for Inference API. API access may be limited.")
        
        # Use direct REST API approach for inference
        try:
            import requests
            import json
            
            # Store API information
            self.api_key = api_key
            self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Create a simple wrapper function that mimics the model interface
            def generate_text(messages):
                try:
                    if isinstance(messages, list) and messages:
                        # Format the input for the inference API
                        if isinstance(messages[-1], dict) and 'content' in messages[-1]:
                            # Handle modern message format
                            prompt = messages[-1]["content"]
                            if isinstance(prompt, list):  # Handle content lists
                                prompt = " ".join([item.get("text", "") for item in prompt if item.get("type") == "text"])
                        else:
                            # Handle string or other formats
                            prompt = str(messages[-1])
                        
                        # Check if the model accepts chat format
                        is_chat_model = any(chat_term in self.model_id.lower() for chat_term in ["chat", "instruct", "llama", "mistral"])
                        
                        if is_chat_model:
                            # Use chat completions API for chat models
                            payload = {
                                "inputs": {
                                    "messages": [
                                        {"role": "user", "content": prompt}
                                    ]
                                },
                                "parameters": {
                                    "max_new_tokens": self.max_new_tokens,
                                    "temperature": self.temperature,
                                    "do_sample": True
                                }
                            }
                        else:
                            # Use text generation API for non-chat models
                            payload = {
                                "inputs": prompt,
                                "parameters": {
                                    "max_new_tokens": self.max_new_tokens,
                                    "temperature": self.temperature,
                                    "do_sample": True
                                }
                            }
                        
                        # Make request to API
                        response = requests.post(self.api_url, headers=self.headers, json=payload)
                        
                        # Handle response
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Different models return different formats
                            if isinstance(result, list) and len(result) > 0:
                                if 'generated_text' in result[0]:
                                    return result[0]['generated_text']
                                else:
                                    return str(result[0])
                            elif isinstance(result, dict):
                                if 'generated_text' in result:
                                    return result['generated_text']
                                elif 'choices' in result and len(result['choices']) > 0:
                                    return result['choices'][0].get('message', {}).get('content', str(result))
                                else:
                                    return str(result)
                            else:
                                return str(result)
                        else:
                            error_msg = f"API error: {response.status_code} - {response.text}"
                            self.logger.error(error_msg)
                            
                            # If it's a payment issue, provide more helpful message
                            if response.status_code == 402:
                                return "Error: This model requires additional payment credits. Please try a smaller model or switch to local execution mode."
                            
                            return f"Error calling Inference API: {error_msg}"
                    else:
                        return "Error: Invalid input format for inference API."
                except Exception as e:
                    self.logger.error(f"Inference API error: {str(e)}")
                    return f"Error calling Inference API: {str(e)}"
            
            # Assign the wrapper as our model
            self.model = generate_text
            self.logger.info(f"Initialized {self.model_id} using direct REST API for inference")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Inference API: {str(e)}")
            raise

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