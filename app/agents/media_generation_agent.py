"""
Media Generation Agent for SagaX1
Agent for generating images from text prompts using Hugging Face spaces
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Callable
from PIL import Image
import io
import base64

from app.agents.base_agent import BaseAgent
from smolagents import Tool, CodeAgent

# Known working image generation spaces
FALLBACK_SPACES = [
    "black-forest-labs/FLUX.1-schnell",  # Fast version
    "black-forest-labs/FLUX.1-dev",      # Higher quality version
    "stabilityai/sdxl",                  # Another alternative
    "runwayml/stable-diffusion-v1-5"     # Last resort fallback
]

class MediaGenerationAgent(BaseAgent):
    """Agent for generating images from text prompts"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the media generation agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                max_tokens: Maximum number of tokens to generate
                temperature: Temperature for generation
                image_space_id: Hugging Face space ID for image generation
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
        self.device = config.get("device", "auto")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        
        # CRITICAL: Override the image_space_id here, regardless of what's in config
        # This ensures we use a working space even if old config is loaded
        config["image_space_id"] = "black-forest-labs/FLUX.1-schnell"
        self.image_space_id = "black-forest-labs/FLUX.1-schnell"  # Use the fast version by default
        
        self.authorized_imports = config.get("authorized_imports", [])
        
        self.image_tool = None
        self.agent = None
        self.is_initialized = False
        
        # Store generated media paths
        self.generated_media = []
    
    def initialize(self) -> None:
        """Initialize the model and agent"""
        if self.is_initialized:
            return
        
        try:
            from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
            
            self.logger.info(f"Initializing image generation agent with model {self.model_id}")
            
            # Try to create the model based on the model_id
            try:
                # First try TransformersModel for local models
                model = TransformersModel(
                    model_id=self.model_id,
                    device_map=self.device,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    trust_remote_code=True
                )
                self.logger.info(f"Using TransformersModel for {self.model_id}")
            except Exception as e:
                self.logger.warning(f"Failed to load model with TransformersModel: {str(e)}")
                
                # Try HfApiModel for API-based models
                try:
                    model = HfApiModel(
                        model_id=self.model_id,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    self.logger.info(f"Using HfApiModel for {self.model_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to load model with HfApiModel: {str(e)}")
                    
                    # Fallback to OpenAI-compatible API
                    try:
                        model = OpenAIServerModel(
                            model_id=self.model_id,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature
                        )
                        self.logger.info(f"Using OpenAIServerModel for {self.model_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load model with OpenAIServerModel: {str(e)}")
                        
                        # Final fallback to LiteLLM
                        model = LiteLLMModel(
                            model_id=self.model_id,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature
                        )
                        self.logger.info(f"Using LiteLLMModel for {self.model_id}")
            
            # Initialize image generation tool
            self._initialize_tools_with_failsafe()
            
            # Create the agent
            self.agent = CodeAgent(
                tools=[self.image_tool],
                model=model,
                additional_authorized_imports=["PIL", "gradio_client"] + self.authorized_imports,
                verbosity_level=1
            )
            
            self.is_initialized = True
            self.logger.info(f"Image generation agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing image generation agent: {str(e)}")
            raise
    
    def _initialize_tools_with_failsafe(self) -> None:
        """Initialize image generation tool with failsafe fallbacks"""
        from smolagents import Tool
        
        # First try the specified space
        self.logger.info(f"Attempting to initialize image tool from {self.image_space_id}")
        
        try:
            self.image_tool = Tool.from_space(
                self.image_space_id,
                name="image_generator",
                description="Generate an image from a text prompt"
            )
            self.logger.info(f"Successfully initialized image tool from {self.image_space_id}")
            return
        except Exception as e:
            self.logger.warning(f"Failed to initialize image tool from {self.image_space_id}: {str(e)}")
        
        # Try each fallback space until one works
        for space in FALLBACK_SPACES:
            if space == self.image_space_id:
                continue  # Skip if it's the same as the one we already tried
                
            self.logger.info(f"Attempting fallback: initializing image tool from {space}")
            try:
                self.image_tool = Tool.from_space(
                    space,
                    name="image_generator",
                    description="Generate an image from a text prompt"
                )
                self.logger.info(f"Successfully initialized image tool from fallback space {space}")
                self.image_space_id = space  # Update the space ID to the one that worked
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize image tool from fallback space {space}: {str(e)}")
        
        # If all fallbacks fail, create a dummy tool that returns an error message
        self.logger.error("All image generation spaces failed to initialize")
        raise RuntimeError("Failed to initialize any image generation space")
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input
        
        Args:
            input_text: Input text for the agent
            callback: Optional callback for streaming responses
            
        Returns:
            Agent output text
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Extract the prompt part after "Generate an image:" if present
            prompt = input_text
            if ":" in input_text:
                prompt = input_text.split(":", 1)[1].strip()
            
            # Log the prompt
            self.logger.info(f"Generating image with prompt: {prompt}")
            
            # Simplified direct tool usage to bypass CodeAgent complexities
            try:
                # Try direct tool usage first (most reliable approach)
                self.logger.info(f"Directly using image_generator tool with prompt: {prompt}")
                image_result = self.image_tool(prompt)
                
                # Save image to temp file if it's a PIL Image
                if isinstance(image_result, Image.Image):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        image_path = temp_file.name
                        image_result.save(image_path)
                        self.generated_media.append(image_path)
                        
                        result_message = f"""
I've generated an image based on your prompt: "{prompt}"

The image has been created using the {self.image_space_id} model.
Image saved to: {image_path}

You can view the image in the display area above and save it using the 'Save Media' button.
"""
                        # Add to history
                        self.add_to_history(input_text, result_message)
                        return result_message
                else:
                    # Handle other return types (like URLs or base64)
                    return f"Image generated successfully using {self.image_space_id}. Result: {str(image_result)}"
                    
            except Exception as direct_error:
                self.logger.warning(f"Direct tool usage failed: {str(direct_error)}. Falling back to agent.")
                
                # Fall back to using the agent
                # Fall back to using the agent
                result = self.agent.run(
                    f"""Generate an image based on this prompt: '{prompt}'
                    Use the image_generator tool to create the image.
                    When complete, combine the description and file location into a single string 
                    like "The image shows [description]. It has been saved to [location]." 
                    and pass that to final_answer()."""
                )
                
                # Add to history
                self.add_to_history(input_text, str(result))
                return str(result)
            
        except Exception as e:
            error_msg = f"Error running image generation agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while generating the image: {error_msg}"
    
    def _clean_old_media(self, max_files: int = 20):
        """Clean up old media files to avoid filling up disk space
        
        Args:
            max_files: Maximum number of files to keep
        """
        # Get temporary files that might have been created
        media_files = self.generated_media
        
        # If we have too many files, delete the oldest ones
        if len(media_files) > max_files:
            files_to_delete = media_files[:-max_files]
            for file_path in files_to_delete:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    self.logger.error(f"Error deleting file {file_path}: {str(e)}")
            
            # Update the list
            self.generated_media = media_files[-max_files:]
    
    def reset(self) -> None:
        """Reset the agent's state"""
        if self.agent:
            # Reset the agent's memory
            self.agent.memory.reset()
        
        self.clear_history()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        return ["image_generation", "prompt_improvement"]
    
    def cleanup(self) -> None:
        """Clean up resources"""
        # Clean up any temporary files
        for file_path in self.generated_media:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                self.logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        self.generated_media = []