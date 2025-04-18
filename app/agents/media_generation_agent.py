"""
Media Generation Agent for sagax1
Agent for generating images and videos from text prompts using Hugging Face spaces
Updated to ignore user-selected model and use direct Space APIs
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
FALLBACK_IMAGE_SPACES = [
    "Efficient-Large-Model/SanaSprint",
    "black-forest-labs/FLUX.1-schnell",  # Fast version
    "black-forest-labs/FLUX.1-dev",      # Higher quality version
    "stabilityai/sdxl",                  # Another alternative
    "runwayml/stable-diffusion-v1-5"     # Last resort fallback
]

# Video generation spaces
VIDEO_SPACES = [
    "SahaniJi/Instant-Video",            # Primary video generation space
    "KingNish/Instant-Video",       # Fallback video space
    "camenduru/text-to-video"            # Last resort fallback
]

class MediaGenerationAgent(BaseAgent):
    """Agent for generating images and videos from text prompts"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the media generation agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID (will be ignored for media generation)
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                max_tokens: Maximum number of tokens to generate
                temperature: Temperature for generation
                image_space_id: Hugging Face space ID for image generation
                video_space_id: Hugging Face space ID for video generation
        """
        super().__init__(agent_id, config)
        
        # These model settings are kept for compatibility but won't affect media generation
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
        self.device = config.get("device", "auto")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        
        # CRITICAL: Override the image_space_id here, regardless of what's in config
        # This ensures we use a working space even if old config is loaded
        config["image_space_id"] = config.get("image_space_id", "black-forest-labs/FLUX.1-schnell")
        self.image_space_id = config.get("image_space_id", "black-forest-labs/FLUX.1-schnell")  # Use the fast version by default
        
        # Video space configuration
        config["video_space_id"] = config.get("video_space_id", "SahaniJi/Instant-Video")
        self.video_space_id = config.get("video_space_id", "SahaniJi/Instant-Video")
        
        self.authorized_imports = config.get("authorized_imports", [])
        
        self.image_tool = None
        self.video_tool = None
        self.agent = None
        self.is_initialized = False
        
        # Store generated media paths
        self.generated_media = []
    
    def initialize(self) -> None:
        """Initialize media generation tools directly without using the language model"""
        if self.is_initialized:
            return
        
        try:
            # Skip model initialization - we don't need it for direct media generation
            self.logger.info("Initializing media generation tools (without language model)")
            
            # Initialize image and video generation tools
            self._initialize_tools_with_failsafe()
            
            # We don't need to create an agent for direct tool usage
            self.is_initialized = True
            self.logger.info(f"Media generation agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing media generation agent: {str(e)}")
            raise
    
    def _initialize_tools_with_failsafe(self) -> None:
        """Initialize image and video generation tools with failsafe fallbacks"""
        from smolagents import Tool
        
        # Initialize image generation tool
        self._initialize_image_tool()
        
        # Initialize video generation tool
        self._initialize_video_tool()
    
    def _initialize_image_tool(self) -> None:
        """Initialize image generation tool with fallbacks"""
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
        for space in FALLBACK_IMAGE_SPACES:
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
        
        # If all fallbacks fail, log error but continue (we might still have video tool)
        self.logger.error("All image generation spaces failed to initialize")
        self.image_tool = None
    
    def _initialize_video_tool(self) -> None:
        """Initialize video generation tool with fallbacks"""
        from smolagents import Tool
        
        # First try the specified space
        self.logger.info(f"Attempting to initialize video tool from {self.video_space_id}")
        
        try:
            self.video_tool = Tool.from_space(
                self.video_space_id,
                name="video_generator",
                description="Generate a video from a text prompt"
            )
            self.logger.info(f"Successfully initialized video tool from {self.video_space_id}")
            return
        except Exception as e:
            self.logger.warning(f"Failed to initialize video tool from {self.video_space_id}: {str(e)}")
        
        # Try each fallback space until one works
        for space in VIDEO_SPACES:
            if space == self.video_space_id:
                continue  # Skip if it's the same as the one we already tried
                
            self.logger.info(f"Attempting fallback: initializing video tool from {space}")
            try:
                self.video_tool = Tool.from_space(
                    space,
                    name="video_generator",
                    description="Generate a video from a text prompt"
                )
                self.logger.info(f"Successfully initialized video tool from fallback space {space}")
                self.video_space_id = space  # Update the space ID to the one that worked
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize video tool from fallback space {space}: {str(e)}")
        
        # If all fallbacks fail, log error but continue (we might still have image tool)
        self.logger.error("All video generation spaces failed to initialize")
        self.video_tool = None
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input - directly calling tools without using a language model
        
        Args:
            input_text: Input text for the agent
            callback: Optional callback for streaming responses
            
        Returns:
            Agent output text
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Determine if this is a request for image or video generation
            is_video_request = False
            if "generate a video" in input_text.lower() or "video:" in input_text.lower():
                is_video_request = True
                if not self.video_tool:
                    return "Sorry, video generation is not available at the moment. The video generation tool could not be initialized."
            elif "generate an image" in input_text.lower() or "image:" in input_text.lower():
                if not self.image_tool:
                    return "Sorry, image generation is not available at the moment. The image generation tool could not be initialized."
            
            # Extract the prompt part after any prefix
            prompt = input_text
            if ":" in input_text:
                prompt = input_text.split(":", 1)[1].strip()
            
            # Log the prompt
            if is_video_request:
                self.logger.info(f"Generating video with prompt: {prompt}")
                return self._generate_video(prompt, callback)
            else:
                self.logger.info(f"Generating image with prompt: {prompt}")
                return self._generate_image(prompt, callback)
            
        except Exception as e:
            error_msg = f"Error running media generation agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while generating the media: {error_msg}"
    
    def _generate_image(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate an image based on the prompt
        
        Args:
            prompt: Text prompt for image generation
            callback: Optional callback for progress updates
            
        Returns:
            Result message with image information
        """
        try:
            # Update progress if callback is provided
            if callback:
                callback("Generating image, please wait...")
            
            # Directly use the image generation tool
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
                    self.add_to_history(f"Generate an image: {prompt}", result_message)
                    return result_message
            else:
                # Handle other return types (like URLs or base64)
                return f"Image generated successfully. Result: {str(image_result)}"
                
        except Exception as direct_error:
            self.logger.error(f"Error in image generation: {str(direct_error)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return f"""
Sorry, I encountered an error while generating the image: {str(direct_error)}

This could be due to:
- The image generation service being temporarily unavailable
- An issue with the prompt
- Connection problems with the Hugging Face space

Please try again with a different prompt or try later when the service might be available.
"""
    
    def _generate_video(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate a video based on the prompt
        
        Args:
            prompt: Text prompt for video generation
            callback: Optional callback for progress updates
            
        Returns:
            Result message with video information
        """
        try:
            # Update progress if callback is provided
            if callback:
                callback("Generating video, this may take a minute...")
            
            # Import required libraries
            from gradio_client import Client
            import os
            
            self.logger.info(f"Generating video with prompt: {prompt}")
            
            # Create a direct client connection to the video space
            client = Client(self.video_space_id)
            self.logger.info(f"Connected to {self.video_space_id}")
            
            # For SahaniJi/Instant-Video, use the specific endpoint and parameters
            # Based on our testing, we know the correct API endpoint and parameters
            api_name = "/instant_video"
            
            # Default parameters 
            base = "Realistic"
            motion = "guoyww/animatediff-motion-lora-zoom-in"
            step = "4"
            
            # Check if the prompt includes specific style instructions
            lower_prompt = prompt.lower()
            if "cartoon" in lower_prompt or "animated" in lower_prompt:
                base = "Cartoon"
            elif "anime" in lower_prompt:
                base = "Anime"
            elif "3d" in lower_prompt:
                base = "3d"
            
            # Check for motion instructions
            if "zoom in" in lower_prompt:
                motion = "guoyww/animatediff-motion-lora-zoom-in"
            elif "zoom out" in lower_prompt:
                motion = "guoyww/animatediff-motion-lora-zoom-out"
            elif "pan left" in lower_prompt:
                motion = "guoyww/animatediff-motion-lora-pan-left"
            elif "pan right" in lower_prompt:
                motion = "guoyww/animatediff-motion-lora-pan-right"
            elif "tilt up" in lower_prompt:
                motion = "guoyww/animatediff-motion-lora-tilt-up"
            elif "tilt down" in lower_prompt:
                motion = "guoyww/animatediff-motion-lora-tilt-down"
            elif "roll clockwise" in lower_prompt:
                motion = "guoyww/animatediff-motion-lora-rolling-clockwise"
            elif "roll counterclockwise" in lower_prompt or "roll anticlockwise" in lower_prompt:
                motion = "guoyww/animatediff-motion-lora-rolling-anticlockwise"
            
            # Log the parameters we're using
            self.logger.info(f"Video generation parameters: prompt='{prompt}', base='{base}', motion='{motion}', step='{step}'")
            
            # Generate the video with the specific parameters
            result = client.predict(
                prompt,
                base,
                motion,
                step,
                api_name=api_name
            )
            
            self.logger.info(f"Video generation result type: {type(result)}")
            
            # The result should be a dictionary with a 'video' key
            video_path = None
            
            if isinstance(result, dict) and 'video' in result:
                video_path = result['video']
                self.logger.info(f"Found video path in result dictionary: {video_path}")
                
                # Add to generated media list for cleanup later
                self.generated_media.append(video_path)
            else:
                self.logger.warning(f"Unexpected result format: {result}")
                
                # If we don't get the expected format, create a temp file
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    video_path = temp_file.name
                    self.logger.info(f"Saving result to temporary file: {video_path}")
                    
                    if isinstance(result, bytes):
                        temp_file.write(result)
                    elif hasattr(result, 'read'):
                        temp_file.write(result.read())
                    elif isinstance(result, str):
                        temp_file.write(result.encode('utf-8'))
                    else:
                        # Just write the string representation
                        temp_file.write(str(result).encode('utf-8'))
                    
                    self.generated_media.append(video_path)
            
            # Check that the video file exists and has a reasonable size
            if video_path and os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                self.logger.info(f"Video file size: {file_size} bytes")
                
                if file_size < 1000:  # Less than 1KB is suspicious
                    self.logger.warning(f"Video file is suspiciously small: {file_size} bytes")
                    return f"""
    The video generation attempt completed, but the resulting file is too small ({file_size} bytes) to be a valid video.

    This usually happens when the video generation service didn't return actual video content. You might want to:
    1. Try a different prompt
    2. Try again later as the service might be experiencing issues
    3. Check if the service has usage limits

    Technical details: Attempted to generate video with prompt: "{prompt}" using {self.video_space_id}.
    """
                
                # Success! Return a message with the video path
                result_message = f"""
    I've generated a video based on your prompt: "{prompt}"

    Video saved to: {video_path}

    You can view the video in your media player and save it using the 'Save Media' button.
    """
                # Add to history
                self.add_to_history(f"Generate a video: {prompt}", result_message)
                return result_message
            else:
                self.logger.error(f"Video file not found at path: {video_path}")
                return f"""
    Sorry, the video generation service ran but didn't produce a valid video file.

    This could be due to:
    - The video generation service being temporarily overloaded
    - An issue with processing your specific prompt
    - Service limitations

    Please try again with a different prompt or try later when the service might be less busy.
    """
            
        except Exception as e:
            self.logger.error(f"Error in video generation: {str(e)}", exc_info=True)
            return f"""
    Sorry, I encountered an error while generating the video: {str(e)}

    This could be due to:
    - The video generation service being temporarily unavailable
    - An issue with the prompt
    - Connection problems with the Hugging Face space

    Please try again with a different prompt or try later when the service might be available.
    """
    
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
        # Clean up old media files
        self._clean_old_media()
        
        self.clear_history()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        capabilities = ["prompt_improvement"]
        
        if self.image_tool:
            capabilities.append("image_generation")
        
        if self.video_tool:
            capabilities.append("video_generation")
            
        return capabilities
    
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