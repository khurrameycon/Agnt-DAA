"""
Code Generation Agent for sagax1
Agent that uses the Hugging Face Inference API to generate code from text prompts
"""

import os
import logging
import traceback
from typing import Dict, Any, List, Optional, Callable
import re

# Imports for the direct Inference API approach
from huggingface_hub import InferenceClient
from app.core.config_manager import ConfigManager

# BaseAgent is still needed
from app.agents.base_agent import BaseAgent

class CodeGenerationAgent(BaseAgent):
    """Agent for generating code from text prompts using direct Inference API"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the code generation agent

        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary (note: model_id selection is bypassed)
        """
        super().__init__(agent_id, config)

        # Get common config (temperature, max_tokens)
        # model_id from config is ignored as per user request
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7) # Default temp for code gen

        # --- Get API Key ---
        self.api_key = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_API_KEY")
        if not self.api_key:
            try:
                # Use ConfigManager to get the key
                config_manager = ConfigManager()
                self.api_key = config_manager.get_hf_api_key()
                if self.api_key:
                    self.logger.info("HF API key successfully retrieved from ConfigManager for CodeGenAgent.")
                else:
                    # Log an error, but allow initialization to proceed if API key is missing,
                    # initialize() will handle the error properly.
                    self.logger.error("HF API key not found in environment variables or config file for CodeGenAgent.")
            except Exception as e:
                self.logger.error(f"Error retrieving HF API key using ConfigManager: {str(e)}")
                self.api_key = None
        # --- End Get API Key ---

        # --- Hardcoded Model ---
        api_provider = self.config.get("api_provider", "huggingface")
        if api_provider == "openai":
            self.target_model = "gpt-4o-mini"
        elif api_provider == "gemini":
            self.target_model = "gemini-2.0-flash-exp"
        elif api_provider == "groq":
            self.target_model = "llama-3.3-70b-versatile"
        else:
            self.target_model = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.logger.info(f"Code Generation Agent configured")
        # --- End Hardcoded Model ---

        # --- Inference Client (initialized later) ---
        self.inference_client = None
        # --- End Inference Client ---

        self.is_initialized = False
        self.generated_code = [] # Keep track of generated snippets

        # Setup logging
        self.logger = logging.getLogger(f"CodeGenAgent-{agent_id}")


    def initialize(self) -> None:
        """Initialize the InferenceClient"""
        if self.is_initialized:
            return

        self.logger.info(f"Initializing InferenceClient for CodeGenAgent {self.agent_id}")
        if not self.api_key:
            self.logger.error("Cannot initialize InferenceClient: Hugging Face API key is missing.")
            # Do not set is_initialized to True
            return

        try:
            # Initialize InferenceClient
            self.inference_client = InferenceClient(
                token=self.api_key # Use token parameter
            )
            self.is_initialized = True
            self.logger.info(f"CodeGenAgent {self.agent_id} initialized successfully with InferenceClient")

        except Exception as e:
            self.logger.error(f"Error initializing InferenceClient for CodeGenAgent: {str(e)}")
            traceback.print_exc()
            # Ensure is_initialized remains False


    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input using API providers or HF Inference API
        
        Args:
            input_text: Input text (prompt) for the agent
            callback: Optional callback for streaming responses
            
        Returns:
            Generated code or error message
        """
        # Get API provider from config
        api_provider = self.config.get("api_provider", "huggingface")
        
        if api_provider in ["openai", "gemini", "groq"]:
            # Use new API providers
            from app.utils.api_providers import APIProviderFactory
            from app.core.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            api_keys = {
                "openai": config_manager.get_openai_api_key(),
                "gemini": config_manager.get_gemini_api_key(),
                "groq": config_manager.get_groq_api_key()
            }
            
            api_key = api_keys.get(api_provider)
            if not api_key:
                error_msg = f"{api_provider.upper()} API key is required for {api_provider} mode"
                self.add_to_history(input_text, error_msg)
                return error_msg
            
            prompt = input_text.strip()
            self.logger.info(f"Generating code using {api_provider.upper()} for prompt: '{prompt[:50]}...'")
            
            if callback:
                callback(f"Generating code with {api_provider.upper()}...")
            
            try:
                provider_instance = APIProviderFactory.create_provider(api_provider, api_key, self.target_model)
                
                # Create a code-focused prompt
                code_prompt = f"Generate clean, well-commented code for the following request:\n\n{prompt}\n\nProvide only the code with appropriate comments:"
                
                messages = [{"content": code_prompt}]
                response = provider_instance.generate(
                    messages, 
                    temperature=self.temperature, 
                    max_tokens=self.max_tokens
                )
                
                self.logger.info(f"Code generation successful with {api_provider.upper()}. Response length: {len(response)}")
                
                # Extract code snippet using the helper
                code_snippet = self._extract_code_from_result(response)
                if code_snippet:
                    self.generated_code.append(code_snippet)
                    # Use the formatted response with the extracted code
                    result_message = self._format_code_response(prompt, code_snippet)
                else:
                    # If no specific code block found, use the whole text but format it
                    result_message = self._format_code_response(prompt, response)
                
                # Add to history
                self.add_to_history(input_text, result_message)
                return result_message
                
            except Exception as e:
                error_msg = f"Error with {api_provider.upper()}: {str(e)}"
                self.logger.error(error_msg)
                self.add_to_history(input_text, error_msg)
                return error_msg
        
        else:
            # Use existing HuggingFace Inference API implementation
            if not self.is_initialized or not self.inference_client:
                # Try to initialize if not already done (e.g., if API key was added after initial attempt)
                self.initialize()
                if not self.is_initialized or not self.inference_client:
                    error_msg = "Code Generation Agent not initialized (possibly missing API key)."
                    self.logger.error(error_msg)
                    return f"Error: {error_msg}"

            prompt = input_text.strip()
            self.logger.info(f"Generating code for Prompt: '{prompt[:50]}...'")

            if callback:
                callback(f"Generating code with Hugging Face API...")

            try:
                # Prepare messages for the chat completions endpoint
                messages = [{"role": "user", "content": prompt}]

                # Make the API Call
                completion = self.inference_client.chat.completions.create(
                    model=self.target_model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else None, # Temp must be > 0 for API
                    # top_p=0.95, # Optional: Add top_p if desired
                    stop=None, # Optional: Add stop sequences if needed, e.g., ["```"]
                    stream=False # Keep stream False for this implementation
                )

                # Extract the response content
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    generated_text = completion.choices[0].message.content
                    self.logger.info(f"Code generation successful. Response length: {len(generated_text)}")

                    # Extract code snippet using the helper
                    code_snippet = self._extract_code_from_result(generated_text)
                    if code_snippet:
                        self.generated_code.append(code_snippet)
                        # Use the formatted response with the extracted code
                        result_message = self._format_code_response(prompt, code_snippet)
                    else:
                        # If no specific code block found, use the whole text but format it
                        result_message = self._format_code_response(prompt, generated_text)

                    # Add to history
                    self.add_to_history(input_text, result_message)
                    return result_message
                else:
                    self.logger.error("Received an unexpected or empty response structure from the API.")
                    self.add_to_history(input_text, "Error: No valid response from API.")
                    return "Error: Received no valid choices from the API."

            except Exception as e:
                error_msg = f"Error during API call model: {str(e)}"
                self.logger.error(error_msg)
                traceback.print_exc()
                # Add to history with error
                self.add_to_history(input_text, f"Error: {error_msg}")
                return f"Sorry, I encountered an error while generating code: {error_msg}"


    def _extract_code_from_result(self, result: str) -> Optional[str]:
        """Extract code blocks from the result using regex.

        Args:
            result: Result string from the agent.

        Returns:
            The first extracted code block, or None if no block is found.
        """
        # Look for markdown code blocks (```python ... ``` or ``` ... ```)
        code_blocks = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", result, re.IGNORECASE)

        if code_blocks:
            # Return the content of the first block found
            return code_blocks[0].strip()

        # If no explicit python blocks, try generic blocks
        generic_blocks = re.findall(r"```([\s\S]*?)```", result)
        if generic_blocks:
             return generic_blocks[0].strip()

        # If no blocks found, return None
        return None


    def _format_code_response(self, prompt: str, code: str) -> str:
        """Format code response with markdown

        Args:
            prompt: Original prompt
            code: Generated code snippet

        Returns:
            Formatted response string
        """
        # Try to determine the language for markdown fencing
        language = self._guess_language(code)

        # Format the response
        return f"""
Based on your prompt: "{prompt}"

Here is the generated code:

```{language}
{code}"""
    
    def _guess_language(self, code: str) -> str:
        """Try to guess the programming language of the code.

        Args:
            code: Code snippet string.

        Returns:
            Lowercase language name (e.g., "python") or empty string if unsure.
        """
        # Simple checks for common languages
        if "def " in code and ":" in code and ("import " in code or "print(" in code or "class " in code):
            return "python"
        elif "function " in code and ("{" in code or "=>" in code or "const " in code or "let " in code):
            return "javascript"
        elif ("public class " in code or "public static void main" in code) and ";" in code:
            return "java"
        elif "#include" in code and ("<" in code and ">" in code or "int main" in code) and ";" in code:
            return "cpp"
        elif "using namespace" in code and "int main" in code and ";" in code:
            return "cpp"
        elif "package main" in code and "func " in code:
            return "go"
        elif "<?php" in code:
            return "php"
        elif "<html" in code or "<!DOCTYPE" in code or "<div>" in code:
            return "html"
        elif "<style>" in code or "{" in code and "}" in code and ":" in code:
            return "css"
        elif "SELECT " in code.upper() and "FROM " in code.upper() and ";" in code:
            return "sql"
        # Add more checks as needed for other languages (Rust, Swift, etc.)
        else:
            # Default to empty string if unsure
            return ""


    def reset(self) -> None:
        """Reset the agent's state (clears history)"""
        self.clear_history()
        self.generated_code = []
        # No agent-specific state (like smolagents memory) to reset anymore


    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has

        Returns:
            List of capability names
        """
        return ["code_generation", "programming_assistance", "api_inference"]