"""
Web Browsing Agent for SagaX1
Agent for browsing the web, searching for information, and visiting webpages
Enhanced with improved UI display based on multiagents.ipynb sample
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
    Enhanced with better UI display capabilities
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the web browsing agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                max_tokens: Maximum number of tokens to generate
                temperature: Temperature for generation
                multi_agent: Whether to use multi-agent architecture
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
        self.device = config.get("device", "auto")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        self.authorized_imports = config.get("authorized_imports", [])
        # Add required imports for web tools
        for import_name in ["requests", "bs4", "json", "re", "os"]:
            if import_name not in self.authorized_imports:
                self.authorized_imports.append(import_name)
                
        self.use_multi_agent = config.get("multi_agent", False)
        
        self.main_agent = None
        self.web_agent = None
        self.manager_agent = None
        self.is_initialized = False
        
        # Store the last raw search results for UI display
        self.last_search_results = ""
    
    def initialize(self) -> None:
        """Initialize the model and agent(s)"""
        if self.is_initialized:
            return
        
        try:
            from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
            
            self.logger.info(f"Initializing web browsing agent with model {self.model_id}")
            
            # Attempt to load the model
            model = self._load_model()

            # Initialize tools
            tools = self._initialize_tools()

            # If multi-agent mode is enabled, initialize multi-agent system
            if self.use_multi_agent:
                self._initialize_multi_agent(model, tools)
            else:
                # Create a single agent for direct search
                self.main_agent = ToolCallingAgent(
                    tools=tools,
                    model=model,
                    max_steps=5,  # Allow several steps for search and optional page visits
                    verbosity_level=2  # Provide detailed output logs
                )
            
            self.is_initialized = True
            self.logger.info(f"Web browsing agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing web browsing agent: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the language model based on configuration"""
        from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
        
        try:
            # First try TransformersModel for local models
            model = TransformersModel(
                model_id=self.model_id,
                device_map=self.device,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                trust_remote_code=True,
                do_sample=True
            )
            self.logger.info(f"Using TransformersModel for {self.model_id}")
            return model
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
                return model
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
                    return model
                except Exception as e:
                    self.logger.warning(f"Failed to load model with OpenAIServerModel: {str(e)}")
                    
                    # Final fallback to LiteLLM
                    model = LiteLLMModel(
                        model_id=self.model_id,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    self.logger.info(f"Using LiteLLMModel for {self.model_id}")
                    return model
    
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
            # Directly use the DuckDuckGoSearchTool for simplicity and reliability
            search_tool = DuckDuckGoSearchTool()
            
            # Perform the search
            search_results = search_tool(query=input_text)
            
            # Clean up the formatting
            # Replace markdown formatting
            formatted_results = f"Search Results for: {input_text}\n\n"
            
            # Process the results to improve formatting
            lines = search_results.split("\n")
            for line in lines:
                # Handle link lines - extract URL and title
                if line.startswith("|") and "](" in line:
                    # Extract title and URL from markdown link format |title](url)
                    parts = line.split("](")
                    if len(parts) >= 2:
                        title = parts[0].replace("|", "").strip()
                        url = parts[1].rstrip(")")
                        formatted_results += f"{title}\n{url}\n\n"
                else:
                    # Regular text line
                    formatted_results += f"{line}\n"
            
            # Add to history
            self.add_to_history(input_text, formatted_results)
            
            # Return just the formatted search results
            return formatted_results
            
        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while searching the web: {error_msg}"
            
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
            "result_formatting"
        ]
        
        if self.use_multi_agent:
            capabilities.append("multi_agent_collaboration")
            capabilities.append("task_planning")
        
        return capabilities