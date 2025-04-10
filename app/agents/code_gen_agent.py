"""
Code Generation Agent for sagax1
Agent that uses Hugging Face spaces to generate code from text prompts
"""

import os
import logging
import tempfile
import time
import traceback
from typing import Dict, Any, List, Optional, Callable

from app.agents.base_agent import BaseAgent
from smolagents import Tool, CodeAgent
from gradio_client import Client

# Known working code generation spaces
CODE_GENERATION_SPACES = [
    "sitammeur/Qwen-Coder-llamacpp",  # Primary code generation space
    "Pradipta01/Code_Generator",       # Fallback code generation space
    "microsoft/CodeX-code-generation", # Another alternative
    "meta/code-llama"                  # Last resort fallback
]

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

class CodeGenerationAgent(BaseAgent):
    """Agent for generating code from text prompts"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the code generation agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                max_tokens: Maximum number of tokens to generate
                temperature: Temperature for generation
                code_space_id: Hugging Face space ID for code generation
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
        self.device = config.get("device", "auto")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        
        # CRITICAL: Override the code_space_id here, regardless of what's in config
        # This ensures we use a working space even if old config is loaded
        config["code_space_id"] = config.get("code_space_id", "sitammeur/Qwen-Coder-llamacpp")
        self.code_space_id = config.get("code_space_id", "sitammeur/Qwen-Coder-llamacpp")
        
        self.authorized_imports = config.get("authorized_imports", [])
        
        self.code_tool = None
        self.agent = None
        self.is_initialized = False
        
        # Store generated code snippets
        self.generated_code = []
        
        # Setup logging
        self.logger = logging.getLogger(f"CodeGenerationAgent-{agent_id}")
    
    def initialize(self) -> None:
        """Initialize the model and agent"""
        if self.is_initialized:
            return
        
        try:
            from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
            
            self.logger.info(f"Initializing code generation agent with model {self.model_id}")
            
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
            
            # Initialize code generation tool directly with our custom implementation
            self._initialize_tools_with_failsafe()
            
            # Create the agent
            self.agent = CodeAgent(
                tools=[self.code_tool],
                model=model,
                additional_authorized_imports=["gradio_client"] + self.authorized_imports,
                verbosity_level=1
            )
            
            self.is_initialized = True
            self.logger.info(f"Code generation agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing code generation agent: {str(e)}")
            traceback.print_exc()
            raise
    
    def _initialize_tools_with_failsafe(self) -> None:
        """Initialize code generation tool with failsafe fallbacks"""
        import time
        
        # First try the specified space
        self.logger.info(f"Attempting to initialize code generation tool from {self.code_space_id}")
        
        try:
            # For the primary space (sitammeur/Qwen-Coder-llamacpp), we handle it with our custom tool
            if "sitammeur/Qwen-Coder" in self.code_space_id:
                self.code_tool = QwenCoderTool(self.code_space_id)
                self.logger.info(f"Successfully initialized Qwen Coder tool from {self.code_space_id}")
                return
            else:
                # For other spaces, use the standard Tool.from_space method
                self.code_tool = Tool.from_space(
                    self.code_space_id,
                    name="code_generator",
                    description="Generate code from a text prompt"
                )
                self.logger.info(f"Successfully initialized code tool from {self.code_space_id}")
                return
        except Exception as e:
            self.logger.warning(f"Failed to initialize code tool from {self.code_space_id}: {str(e)}")
            traceback.print_exc()
        
        # Try each fallback space until one works
        for space in CODE_GENERATION_SPACES:
            if space == self.code_space_id:
                continue  # Skip if it's the same as the one we already tried
                
            self.logger.info(f"Attempting fallback: initializing code tool from {space}")
            try:
                # Special handling for Qwen-Coder space
                if "sitammeur/Qwen-Coder" in space:
                    self.code_tool = QwenCoderTool(space)
                    self.logger.info(f"Successfully initialized Qwen Coder tool from {space}")
                else:
                    # For other spaces, use the standard Tool.from_space method
                    self.code_tool = Tool.from_space(
                        space,
                        name="code_generator",
                        description="Generate code from a text prompt"
                    )
                    self.logger.info(f"Successfully initialized code tool from fallback space {space}")
                
                self.code_space_id = space  # Update the space ID to the one that worked
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize code tool from fallback space {space}: {str(e)}")
                traceback.print_exc()
                # Add a small delay before trying the next space to avoid rate limiting
                time.sleep(1)
        
        # If all fallbacks fail, create a dummy tool as a last resort
        self.logger.error("All code generation spaces failed to initialize")
        
        # Create a dummy tool as a last resort
        self.code_tool = self._create_dummy_tool()
    
    def _create_dummy_tool(self):
        """Create a dummy tool that returns an error message
        
        Returns:
            A dummy tool function
        """
        class DummyTool(Tool):
            name = "code_generator"
            description = "Generate code from a text prompt"
            inputs = {
                "prompt": {
                    "type": "string", 
                    "description": "Text prompt for code generation"
                }
            }
            output_type = "string"
            
            def forward(self, prompt):
                return f"Failed to initialize any code generation space. Unable to generate code for: {prompt}"
        
        return DummyTool()
    
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
            # Clean prompt for code generation
            prompt = input_text.strip()
            
            # Log the prompt
            self.logger.info(f"Generating code with prompt: {prompt}")
            
            # Update progress if callback is provided
            if callback:
                callback("Generating code...")
            
            # Try direct tool usage first (more reliable)
            try:
                # Direct tool usage approach
                self.logger.info(f"Directly using code_generator tool with prompt: {prompt}")
                code_result = self.code_tool(prompt)
                
                # Extract the code from the result
                code_snippet = self._extract_code_from_result(code_result)
                
                # Store the generated code
                if code_snippet:
                    self.generated_code.append(code_snippet)
                
                # Format the response
                result_message = self._format_code_response(prompt, code_snippet)
                
                # Add to history
                self.add_to_history(input_text, result_message)
                return result_message
                
            except Exception as direct_error:
                self.logger.warning(f"Direct tool usage failed: {str(direct_error)}. Falling back to agent.")
                traceback.print_exc()
                
                # Fall back to using the agent
                result = self.agent.run(
                    f"""Generate code based on this prompt: '{prompt}'
                    Use the code_generator tool to create the code.
                    When complete, format the code response with markdown code blocks 
                    and pass that to final_answer()."""
                )
                
                # Add to history
                self.add_to_history(input_text, str(result))
                return str(result)
            
        except Exception as e:
            error_msg = f"Error running code generation agent: {str(e)}"
            self.logger.error(error_msg)
            traceback.print_exc()
            return f"Sorry, I encountered an error while generating code: {error_msg}"
    
    def _extract_code_from_result(self, result) -> str:
        """Extract code from the tool result
        
        Args:
            result: Result from the code generation tool
            
        Returns:
            Extracted code snippet
        """
        # Check the type of result
        if isinstance(result, str):
            # The result is already a string, check if it contains code blocks
            if "```" in result:
                # Extract code from markdown code blocks
                import re
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", result, re.DOTALL)
                if code_blocks:
                    return code_blocks[0].strip()
                
            # Return the entire string as code
            return result.strip()
        
        # If it's a complex object, convert to string
        return str(result).strip()
    
    def _format_code_response(self, prompt: str, code: str) -> str:
        """Format code response with markdown
        
        Args:
            prompt: Original prompt
            code: Generated code
            
        Returns:
            Formatted response
        """
        # Try to determine the language from the code
        language = self._guess_language(code)
        
        # Format the response
        return f"""
Here's the code generated for your prompt: "{prompt}"

```{language}
{code}
```.
"""
    
    def _guess_language(self, code: str) -> str:
        """Try to guess the programming language of the code
        
        Args:
            code: Code snippet
            
        Returns:
            Language name or empty string
        """
        # Check for language-specific patterns
        if "def " in code and ("import " in code or "print(" in code):
            return "python"
        elif "function " in code and ("{" in code or "=>" in code):
            return "javascript"
        elif "public class " in code or "public static void main" in code:
            return "java"
        elif "#include" in code and (("<" in code and ">" in code) or "int main" in code):
            return "cpp"
        elif "using namespace" in code or "int main" in code:
            return "cpp"
        elif "package main" in code and "func " in code:
            return "go"
        elif "<?php" in code:
            return "php"
        elif "<html" in code or "<!DOCTYPE" in code:
            return "html"
        elif "SELECT " in code.upper() and "FROM " in code.upper():
            return "sql"
        else:
            # Default to empty string if we can't determine
            return ""
    
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
        return ["code_generation", "programming_assistance"]