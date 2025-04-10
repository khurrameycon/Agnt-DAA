"""
QwenCoderTool for sagax1
Provides a Tool for generating code using the Qwen Coder model
"""

from smolagents import Tool
from gradio_client import Client
import logging

class QwenCoderTool(Tool):
    """Tool for generating code using the Qwen Coder model"""
    
    name = "code_generator"
    description = "Generate code from a text prompt"
    inputs = {
        "prompt": {
            "type": "string", 
            "description": "Text prompt for code generation"
        }
    }
    output_type = "string"
    
    def __init__(self, space_id):
        """Initialize the Qwen Coder tool
        
        Args:
            space_id: Hugging Face space ID
        """
        # Initialize the Tool parent class
        super().__init__()
        
        self.client = Client(space_id)
        self.space_id = space_id
        self.logger = logging.getLogger(__name__)
    
    def forward(self, prompt):
        """Generate code using the Qwen Coder model
        
        Args:
            prompt: Input prompt for code generation
            
        Returns:
            Generated code
        """
        try:
            # Use the /chat API endpoint with the required parameters
            result = self.client.predict(
                prompt,  # message parameter
                "Qwen2.5-Coder-0.5B-Instruct-Q6_K.gguf",  # model parameter
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that generates code.",  # system prompt
                1024,  # max_length
                0.7,   # temperature
                0.95,  # top_p
                40,    # frequency_penalty
                1.1,   # presence_penalty
                api_name="/chat"
            )
            
            # Extract code from the response
            if isinstance(result, dict) and "response" in result:
                return result["response"]
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"Error generating code with Qwen Coder: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"Failed to generate code: {str(e)}"