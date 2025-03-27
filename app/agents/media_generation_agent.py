"""
Media Generation Agent for SagaX1
Agent for generating images and videos from text prompts using Hugging Face spaces
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

class ImageGenerationTool(Tool):
    """Tool for generating images from text prompts using Hugging Face spaces"""
    
    name = "image_generator"
    description = "Generate an image from a text prompt"
    inputs = {
        "prompt": {
            "type": "string", 
            "description": "Text prompt describing the image to generate"
        },
        "negative_prompt": {
            "type": "string", 
            "description": "Text describing what you don't want in the image (optional)"
        },
        "style": {
            "type": "string", 
            "description": "Style for the image generation (optional)"
        }
    }
    output_type = "image"
    
    def __init__(self, space_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """Initialize the image generation tool
        
        Args:
            space_id: ID of the Hugging Face space to use for image generation
        """
        super().__init__()
        self.space_id = space_id
        self.logger = logging.getLogger(__name__)
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the gradio client"""
        if self._client is None:
            try:
                from gradio_client import Client
                self._client = Client(self.space_id)
                self.logger.info(f"Connected to Hugging Face space: {self.space_id}")
            except Exception as e:
                self.logger.error(f"Error connecting to Hugging Face space: {str(e)}")
                raise
        return self._client
    
    def forward(self, prompt: str, negative_prompt: str = "", style: str = "") -> Image.Image:
        """Generate an image from a prompt
        
        Args:
            prompt: Text prompt describing the image to generate
            negative_prompt: Text describing what you don't want in the image
            style: Style for the image generation
            
        Returns:
            Generated image
        """
        try:
            # Prepare inputs based on the space's requirements
            # This may need to be adjusted for different spaces
            inputs = [prompt]
            if negative_prompt:
                inputs.append(negative_prompt)
            if style:
                inputs.append(style)
            
            # Call the space API
            result = self.client.predict(*inputs)
            
            # Process the result based on the space's output format
            # This may return a path to an image file, an image object, or a base64 string
            if isinstance(result, str):
                if result.startswith("data:image"):
                    # Handle base64 encoded image
                    base64_data = result.split(",")[1]
                    image_data = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(image_data))
                elif os.path.isfile(result):
                    # Handle file path
                    return Image.open(result)
            elif isinstance(result, Image.Image):
                # Already an image
                return result
            else:
                # Try to convert to an image if possible
                try:
                    return Image.open(io.BytesIO(result))
                except:
                    self.logger.error(f"Unexpected result type from space: {type(result)}")
                    # Create a dummy image with error message
                    img = Image.new('RGB', (400, 100), color='red')
                    return img
            
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            # Create a dummy image with error message
            img = Image.new('RGB', (400, 100), color='red')
            return img

class VideoGenerationTool(Tool):
    """Tool for generating videos from text prompts using Hugging Face spaces"""
    
    name = "video_generator"
    description = "Generate a video from a text prompt"
    inputs = {
        "prompt": {
            "type": "string", 
            "description": "Text prompt describing the video to generate"
        },
        "duration": {
            "type": "number", 
            "description": "Duration of the video in seconds (optional)"
        }
    }
    output_type = "string"  # We return a path to the video file
    
    def __init__(self, space_id: str = "damo-vilab/text-to-video-ms"):
        """Initialize the video generation tool
        
        Args:
            space_id: ID of the Hugging Face space to use for video generation
        """
        super().__init__()
        self.space_id = space_id
        self.logger = logging.getLogger(__name__)
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the gradio client"""
        if self._client is None:
            try:
                from gradio_client import Client
                self._client = Client(self.space_id)
                self.logger.info(f"Connected to Hugging Face space: {self.space_id}")
            except Exception as e:
                self.logger.error(f"Error connecting to Hugging Face space: {str(e)}")
                raise
        return self._client
    
    def forward(self, prompt: str, duration: float = 3.0) -> str:
        """Generate a video from a prompt
        
        Args:
            prompt: Text prompt describing the video to generate
            duration: Duration of the video in seconds
            
        Returns:
            Path to the generated video file
        """
        try:
            # Call the space API
            result = self.client.predict(prompt, duration)
            
            # Process the result based on the space's output format
            # This may return a path to a video file or a file object
            if isinstance(result, str) and os.path.isfile(result):
                return result
            else:
                # Create a temporary file to save the video
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Try to save the result to the temporary file
                try:
                    if hasattr(result, 'read'):
                        # If result is a file-like object
                        with open(temp_path, 'wb') as f:
                            f.write(result.read())
                    elif isinstance(result, bytes):
                        # If result is bytes
                        with open(temp_path, 'wb') as f:
                            f.write(result)
                    else:
                        self.logger.error(f"Unexpected result type from space: {type(result)}")
                        return "Error: Unable to process video result"
                    
                    return temp_path
                except Exception as e:
                    self.logger.error(f"Error saving video: {str(e)}")
                    return f"Error generating video: {str(e)}"
            
        except Exception as e:
            self.logger.error(f"Error generating video: {str(e)}")
            return f"Error generating video: {str(e)}"

class MediaGenerationAgent(BaseAgent):
    """Agent for generating images and videos from text prompts"""
    
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
                video_space_id: Hugging Face space ID for video generation
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3-8B-Instruct")
        self.device = config.get("device", "auto")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.image_space_id = config.get("image_space_id", "stabilityai/stable-diffusion-xl-base-1.0")
        self.video_space_id = config.get("video_space_id", "damo-vilab/text-to-video-ms")
        self.authorized_imports = config.get("authorized_imports", [])
        
        self.image_tool = None
        self.video_tool = None
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
            
            self.logger.info(f"Initializing media generation agent with model {self.model_id}")
            
            # Try to create the model based on the model_id
            try:
                # First try TransformersModel for local models
                model = TransformersModel(
                    model_id=self.model_id,
                    device_map=self.device,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
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
            
            # Initialize tools
            self.image_tool = ImageGenerationTool(space_id=self.image_space_id)
            self.video_tool = VideoGenerationTool(space_id=self.video_space_id)
            
            # Create the agent
            self.agent = CodeAgent(
                tools=[self.image_tool, self.video_tool],
                model=model,
                system_prompt="""You are a media generation agent that can create images and videos from text prompts.
You have access to these tools:
- image_generator: Generate an image from a text prompt
- video_generator: Generate a video from a text prompt

For creating images, focus on:
- Providing clear, detailed descriptions
- Specifying style, mood, lighting, and composition
- Using negative prompts to exclude unwanted elements

For creating videos, focus on:
- Describing the scene and action clearly
- Keeping prompts concise but descriptive
- Specifying duration when needed

Always return the generated media to the user.""",
                additional_authorized_imports=["PIL", "gradio_client"] + self.authorized_imports,
                verbosity_level=1
            )
            
            self.is_initialized = True
            self.logger.info(f"Media generation agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing media generation agent: {str(e)}")
            raise
    
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
            # Enhance the prompt for media generation
            enhanced_prompt = f"""
You are a media generation agent that can create images and videos from text prompts.
You have access to image and video generation tools.

USER REQUEST: {input_text}

Analyze the user's request and determine if they want an image or video. Then:
1. Improve the prompt to create high-quality media
2. Use the appropriate tool to generate the requested media
3. Return the result to the user with an explanation
"""
            
            # Run the agent
            result = self.agent.run(enhanced_prompt)
            
            # Add to history
            self.add_to_history(input_text, str(result))
            
            # Check if we need to clean up old media files
            self._clean_old_media()
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Error running media generation agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while generating media: {error_msg}"
    
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
        return ["image_generation", "video_generation", "prompt_improvement"]
    
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