"""
Local Model Agent for SagaX1
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
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3-8B-Instruct")
        self.device = config.get("device", "auto")
        self.max_new_tokens = config.get("max_tokens", 2048)  # Changed to max_new_tokens
        self.temperature = config.get("temperature", 0.1)
        self.authorized_imports = config.get("authorized_imports", [])
        
        self.model = None
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the model"""
        if self.is_initialized:
            return
            
        # Download the model if needed
        self._ensure_model_downloaded()
        
        # Create the model
        self.logger.info(f"Initializing model {self.model_id}")
        
        try:
            # Try to create the model with TransformersModel
            self.model = TransformersModel(
                model_id=self.model_id,
                device_map=self.device,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                trust_remote_code=True,
                do_sample=True  # Add this to fix the temperature warning
            )
            
            self.is_initialized = True
            self.logger.info(f"Model {self.model_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing model with TransformersModel: {str(e)}")
            self._initialize_with_fallbacks()
    
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