"""
Web Browsing Agent for sagax1
Agent for browsing the web, searching for information, and visiting webpages
Updated to remove HF dependency and use only external API providers
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Union

from app.agents.base_agent import BaseAgent
from smolagents import (
    Tool, 
    DuckDuckGoSearchTool, 
    VisitWebpageTool, 
    CodeAgent,
    ToolCallingAgent
)

class FormattedSearchResults(Tool):
    """Tool for formatting search results into a more UI-friendly format"""
    
    name = "format_results"
    description = "Format search and webpage results for better display in the UI"
    inputs = {
        "content": {
            "type": "string", 
            "description": "Raw search results or webpage content to format"
        }
    }
    output_type = "string"
    
    def forward(self, content: str) -> str:
        """Format content for better display
        
        Args:
            content: Raw content to format
            
        Returns:
            Formatted content
        """
        try:
            # Check if the content is from a web search
            if "Web search results:" in content:
                # Format search results more attractively
                lines = content.split("\n")
                formatted = "# Search Results\n\n"
                
                current_result = None
                results = []
                
                for line in lines:
                    if line.startswith("[") and "]" in line:
                        # New result title
                        if current_result:
                            results.append(current_result)
                        current_result = {"title": line, "content": []}
                    elif current_result and line.strip():
                        current_result["content"].append(line)
                
                # Add the last result
                if current_result:
                    results.append(current_result)
                
                # Format results in a more structured way
                for i, result in enumerate(results):
                    title = result["title"]
                    content = "\n".join(result["content"])
                    formatted += f"## Result {i+1}\n{title}\n\n{content}\n\n"
                
                return formatted
            
            # Check if it's webpage content
            elif "Error fetching the webpage:" not in content and len(content) > 500:
                # Simplify lengthy webpage content
                # Extract what seems to be the main content
                paragraphs = [p for p in content.split("\n\n") if len(p) > 100]
                
                if paragraphs:
                    formatted = "# Webpage Content\n\n"
                    formatted += "\n\n".join(paragraphs[:5])  # First 5 substantial paragraphs
                    if len(paragraphs) > 5:
                        formatted += "\n\n... (content continues) ..."
                    return formatted
            
            # Default: return content as is
            return content
            
        except Exception as e:
            return f"Error formatting content: {str(e)}"

class WebBrowsingAgent(BaseAgent):
    """Agent for browsing the web, searching for information, and visiting webpages
    Updated to use only external API providers (OpenAI, Gemini, Groq)
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the web browsing agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Model ID for the API provider
                api_provider: API provider (openai, gemini, groq)
                max_tokens: Maximum number of tokens to generate
                temperature: Temperature for generation
                multi_agent: Whether to use multi-agent architecture
        """
        super().__init__(agent_id, config)
        
        # Get API provider and model configuration
        self.api_provider = config.get("api_provider", "groq")  # Default to Groq
        self.model_id = config.get("model_id", self._get_default_model())
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        self.authorized_imports = config.get("authorized_imports", [])
        
        # Add required imports for web tools
        for import_name in ["requests", "bs4", "json", "re", "os"]:
            if import_name not in self.authorized_imports:
                self.authorized_imports.append(import_name)
        
        # Multi-agent architecture setting
        self.use_multi_agent = config.get("multi_agent", False)
        
        # Initialize components
        self.main_agent = None
        self.web_agent = None
        self.manager_agent = None
        self.model = None
        self.is_initialized = False
        
        # Store the last raw search results for UI display
        self.last_search_results = ""
    
    def _get_default_model(self):
        """Get default model based on API provider"""
        default_models = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash-exp",
            "groq": "llama-3.3-70b-versatile"
        }
        return default_models.get(self.api_provider, "llama-3.3-70b-versatile")
    
    def initialize(self) -> None:
        """Initialize the model and agent(s) using external API providers"""
        if self.is_initialized:
            return
        
        try:
            self.logger.info(f"Initializing web browsing agent with {self.api_provider} API")
            
            # Initialize the LLM model using API providers
            self._initialize_api_model()
            
            # Initialize tools
            tools = self._initialize_tools()

            # If multi-agent mode is enabled, initialize multi-agent system
            if self.use_multi_agent:
                self._initialize_multi_agent(self.model, tools)
            else:
                # Create a single agent for direct search
                self.main_agent = ToolCallingAgent(
                    tools=tools,
                    model=self.model,
                    max_steps=5,  # Allow several steps for search and optional page visits
                    verbosity_level=2  # Provide detailed output logs
                )
            
            self.is_initialized = True
            self.logger.info(f"Web browsing agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing web browsing agent: {str(e)}")
            raise
    
    def _initialize_api_model(self):
        """Initialize the model using external API providers (OpenAI, Gemini, Groq)"""
        from app.utils.api_providers import APIProviderFactory
        from app.core.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        
        # Get API key based on provider
        api_keys = {
            "openai": config_manager.get_openai_api_key(),
            "gemini": config_manager.get_gemini_api_key(),
            "groq": config_manager.get_groq_api_key()
        }
        
        api_key = api_keys.get(self.api_provider)
        if not api_key:
            raise ValueError(f"{self.api_provider.upper()} API key is required for web browsing agent")
        
        # Create API provider instance
        self.api_provider_instance = APIProviderFactory.create_provider(
            self.api_provider, api_key, self.model_id
        )
        
        # Create wrapper function for smolagents compatibility
        def generate_text(messages):
            return self.api_provider_instance.generate(
                messages, 
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        self.model = generate_text
        self.logger.info(f"Initialized {self.api_provider.upper()} API with model {self.model_id}")
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize tools for the agent"""
        return [
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
            FormattedSearchResults()
        ]
    
    def _initialize_multi_agent(self, model, tools: List[Tool]) -> None:
        """Initialize the multi-agent setup following the notebook approach
        
        Args:
            model: Language model to use
            tools: List of tools to use
        """
        # Create a web agent that handles search and browsing (using ToolCallingAgent)
        self.web_agent = ToolCallingAgent(
            tools=tools,
            model=model,
            max_steps=10,  # Allow for more steps as web search may require several iterations
            name="web_search_agent",
            description="Runs web searches and visits webpages for you.",
            verbosity_level=2  # Provide detailed output logs
        )
        
        # Create a manager agent that handles the overall task (using CodeAgent)
        self.manager_agent = CodeAgent(
            tools=[],  # No direct tools, will use the web_agent
            model=model,
            managed_agents=[self.web_agent],  # This is how we connect the agents
            additional_authorized_imports=self.authorized_imports,
            verbosity_level=2  # Provide detailed output logs
        )
    
    def generate_summary_with_llm(self, formatted_results: str, query: str) -> str:
        """Generate an intelligent summary of search results using the LLM
        
        Args:
            formatted_results: Formatted search results
            query: Original search query
            
        Returns:
            LLM-generated summary with relevant URLs
        """
        self.logger.info(f"Generating summary for query: {query}")
        
        # Create a prompt for the LLM
        prompt = f"""You are an intelligent web search assistant that provides helpful, accurate, and concise summaries.

I searched for: "{query}"

Here are the search results:

{formatted_results}

Please provide a comprehensive summary of these results that directly answers my query. 
Include the most relevant information and mention 2-3 specific sources with their URLs.
Format your response in a clear, easy-to-read way with sections and bullet points where appropriate.
"""

        # Format the input in the format expected by the model
        messages = [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Call the model (API-based)
        try:
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
            
            return result_text
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            self.logger.error(error_msg)
            return f"""Failed to generate a summary with the LLM. Here are the raw search results:

{formatted_results}

Error: {error_msg}"""
    
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
            # Update progress if callback is provided
            if callback:
                callback("Searching the web for information...")
            
            # Use the DuckDuckGoSearchTool to get search results
            search_tool = DuckDuckGoSearchTool()
            
            # Perform the search
            search_results = search_tool(query=input_text)
            
            # Store the raw search results
            self.last_search_results = search_results
            
            # Format the search results
            formatter = FormattedSearchResults()
            formatted_results = formatter(search_results)
            
            # Update progress if callback is provided
            if callback:
                callback("Processing search results and generating summary...")
            
            # Generate intelligent summary with the LLM
            summary = self.generate_summary_with_llm(formatted_results, input_text)
            
            # Add to history
            self.add_to_history(input_text, summary)
            
            # Return the summary
            return summary
            
        except Exception as e:
            error_msg = f"Error running web browsing agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while browsing the web: {error_msg}"
    
    def reset(self) -> None:
        """Reset the agent's state"""
        if self.use_multi_agent:
            if self.web_agent:
                self.web_agent.memory.reset()
            if self.manager_agent:
                self.manager_agent.memory.reset()
        elif self.main_agent:
            self.main_agent.memory.reset()
        
        self.clear_history()
        self.last_search_results = ""
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        capabilities = [
            "web_search", 
            "web_browsing", 
            "information_retrieval",
            "content_extraction",
            "result_formatting",
            "intelligent_summarization"
        ]
        
        if self.use_multi_agent:
            capabilities.append("multi_agent_collaboration")
            capabilities.append("task_planning")
        
        # Add API provider capability
        capabilities.append(f"{self.api_provider}_api")
        
        return capabilities