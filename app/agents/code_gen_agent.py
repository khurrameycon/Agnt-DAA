"""
Code Generation Agent for sagax1
Agent that uses external API providers to generate code from text prompts
Updated with Anthropic preference and fallback logic
"""

import os
import logging
import traceback
from typing import Dict, Any, List, Optional, Callable
import re

from app.core.config_manager import ConfigManager
from app.agents.base_agent import BaseAgent

class CodeGenerationAgent(BaseAgent):
    """Agent for generating code from text prompts using external API providers
    Prefers Anthropic API with fallback to other providers"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the code generation agent

        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
        """
        super().__init__(agent_id, config)

        # Get configuration
        self.api_provider = config.get("api_provider", "anthropic")  # Default to Anthropic
        self.model_id = config.get("model_id", self._get_default_model())
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)  # Default temp for code gen

        # Initialize components
        self.api_provider_instance = None
        self.is_initialized = False
        self.generated_code = []  # Keep track of generated snippets

        # Setup logging
        self.logger = logging.getLogger(f"CodeGenAgent-{agent_id}")
        
        self.logger.info(f"Code Generation Agent configured with {self.api_provider} API")

    def _get_default_model(self):
        """Get default model based on API provider"""
        default_models = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash-exp",
            "groq": "llama-3.3-70b-versatile"
        }
        return default_models.get(self.api_provider, "claude-sonnet-4-20250514")

    def _get_available_providers_with_keys(self):
        """Get list of available providers that have API keys configured
        
        Returns:
            List of tuples (provider_name, api_key, default_model)
        """
        config_manager = ConfigManager()
        
        providers = [
            ("anthropic", config_manager.get_anthropic_api_key(), "claude-sonnet-4-20250514"),
            ("openai", config_manager.get_openai_api_key(), "gpt-4o-mini"),
            ("gemini", config_manager.get_gemini_api_key(), "gemini-2.0-flash-exp"),
            ("groq", config_manager.get_groq_api_key(), "llama-3.3-70b-versatile")
        ]
        
        # Return only providers with valid API keys
        available = [(name, key, model) for name, key, model in providers if key]
        return available

    def initialize(self) -> None:
        """Initialize the API provider with fallback logic"""
        if self.is_initialized:
            return

        self.logger.info(f"Initializing CodeGenAgent {self.agent_id}")
        
        # Try to initialize with the preferred provider first
        if self._try_initialize_provider(self.api_provider, self.model_id):
            self.is_initialized = True
            self.logger.info(f"CodeGenAgent {self.agent_id} initialized with {self.api_provider}")
            return
        
        # If preferred provider failed, try fallback providers
        self.logger.warning(f"Failed to initialize with {self.api_provider}, trying fallback providers...")
        
        available_providers = self._get_available_providers_with_keys()
        
        # Remove the already-tried provider from the list
        available_providers = [p for p in available_providers if p[0] != self.api_provider]
        
        for provider_name, api_key, default_model in available_providers:
            self.logger.info(f"Trying fallback provider: {provider_name}")
            if self._try_initialize_provider(provider_name, default_model):
                self.api_provider = provider_name  # Update to working provider
                self.model_id = default_model
                self.is_initialized = True
                self.logger.info(f"CodeGenAgent {self.agent_id} initialized with fallback provider {provider_name}")
                return
        
        # If no providers work, raise an error
        raise RuntimeError(
            "No API providers available for Code Generation. Please configure at least one API key "
            "(Anthropic, OpenAI, Gemini, or Groq) in Settings."
        )

    def _try_initialize_provider(self, provider_name: str, model_id: str) -> bool:
        """Try to initialize a specific API provider
        
        Args:
            provider_name: Name of the provider (anthropic, openai, gemini, groq)
            model_id: Model ID to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from app.utils.api_providers import APIProviderFactory
            from app.core.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            
            # Get API key for the provider
            api_keys = {
                "anthropic": config_manager.get_anthropic_api_key(),
                "openai": config_manager.get_openai_api_key(),
                "gemini": config_manager.get_gemini_api_key(),
                "groq": config_manager.get_groq_api_key()
            }
            
            api_key = api_keys.get(provider_name)
            if not api_key:
                self.logger.warning(f"No API key found for {provider_name}")
                return False
            
            # Create provider instance
            self.api_provider_instance = APIProviderFactory.create_provider(
                provider_name, api_key, model_id
            )
            
            self.logger.info(f"Successfully initialized {provider_name} provider")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {provider_name} provider: {str(e)}")
            return False

    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input using API providers with fallback logic
        
        Args:
            input_text: Input text (prompt) for the agent
            callback: Optional callback for streaming responses
            
        Returns:
            Generated code or error message
        """
        if not self.is_initialized:
            try:
                self.initialize()
            except RuntimeError as e:
                return str(e)

        prompt = input_text.strip()
        self.logger.info(f"Generating code using {self.api_provider.upper()} for prompt: '{prompt[:50]}...'")

        if callback:
            callback(f"Generating code with {self.api_provider.upper()}...")

        try:
            # Create a code-focused prompt
            code_prompt = f"""Generate clean, well-commented code for the following request:

{prompt}

Requirements:
- Provide complete, working code
- Include helpful comments explaining key sections
- Follow best practices for the language
- Make the code readable and maintainable

Code:"""

            messages = [{"content": code_prompt}]
            response = self.api_provider_instance.generate(
                messages, 
                temperature=self.temperature, 
                max_tokens=self.max_tokens
            )

            self.logger.info(f"Code generation successful with {self.api_provider.upper()}. Response length: {len(response)}")

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
            # Try fallback providers if the current one fails
            error_msg = f"Error with {self.api_provider.upper()}: {str(e)}"
            self.logger.error(error_msg)
            
            # Try to switch to a working provider
            if self._try_fallback_provider(prompt, callback):
                # If fallback succeeded, the response was already generated
                return self._last_successful_response
            
            # If all providers failed
            self.add_to_history(input_text, f"Error: {error_msg}")
            return f"Sorry, I encountered an error while generating code: {error_msg}"

    def _try_fallback_provider(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> bool:
        """Try fallback providers when the current one fails
        
        Args:
            prompt: The original prompt
            callback: Optional callback for progress updates
            
        Returns:
            True if a fallback provider worked, False otherwise
        """
        available_providers = self._get_available_providers_with_keys()
        
        # Remove the current (failed) provider
        available_providers = [p for p in available_providers if p[0] != self.api_provider]
        
        for provider_name, api_key, default_model in available_providers:
            try:
                self.logger.info(f"Trying fallback provider: {provider_name}")
                if callback:
                    callback(f"Retrying with {provider_name.upper()}...")
                
                if self._try_initialize_provider(provider_name, default_model):
                    # Update current provider
                    self.api_provider = provider_name
                    self.model_id = default_model
                    
                    # Try to generate with the new provider
                    code_prompt = f"""Generate clean, well-commented code for the following request:

{prompt}

Requirements:
- Provide complete, working code
- Include helpful comments explaining key sections
- Follow best practices for the language
- Make the code readable and maintainable

Code:"""
                    
                    messages = [{"content": code_prompt}]
                    response = self.api_provider_instance.generate(
                        messages, 
                        temperature=self.temperature, 
                        max_tokens=self.max_tokens
                    )
                    
                    # Format the response
                    code_snippet = self._extract_code_from_result(response)
                    if code_snippet:
                        self.generated_code.append(code_snippet)
                        result_message = self._format_code_response(prompt, code_snippet)
                    else:
                        result_message = self._format_code_response(prompt, response)
                    
                    # Store successful response
                    self._last_successful_response = result_message
                    self.add_to_history(prompt, result_message)
                    
                    self.logger.info(f"Successfully switched to {provider_name} provider")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Fallback provider {provider_name} also failed: {str(e)}")
                continue
        
        return False

    def _extract_code_from_result(self, result: str) -> Optional[str]:
        """Extract code blocks from the result using regex.

        Args:
            result: Result string from the agent.

        Returns:
            The first extracted code block, or None if no block is found.
        """
        # Look for markdown code blocks (```python ... ``` or ``` ... ```)
        code_blocks = re.findall(r"```(?:python|py|javascript|js|java|cpp|c\+\+|go|rust|php|ruby|swift|kotlin|typescript|ts|sql|html|css|bash|shell|sh)?\s*([\s\S]*?)```", result, re.IGNORECASE)

        if code_blocks:
            # Return the content of the first block found
            return code_blocks[0].strip()

        # If no explicit language blocks, try generic blocks
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
        return f"""Based on your prompt: "{prompt}"

Here is the generated code:

```{language}
{code}
```

Generated using {self.api_provider.upper()} API with model {self.model_id}."""

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
        else:
            # Default to empty string if unsure
            return ""

    def reset(self) -> None:
        """Reset the agent's state (clears history)"""
        self.clear_history()
        self.generated_code = []

    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has

        Returns:
            List of capability names
        """
        capabilities = [
            "code_generation", 
            "programming_assistance", 
            f"{self.api_provider}_api"
        ]
        
        # Add fallback capability info
        available_providers = self._get_available_providers_with_keys()
        if len(available_providers) > 1:
            capabilities.append("multi_provider_fallback")
        
        return capabilities