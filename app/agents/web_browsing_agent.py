"""
Web Browsing Agent for SagaX1
Agent for browsing the web, searching for information, and visiting webpages
Based on the multi-agent architecture from the notebook
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

class ExtractWebContentTool(Tool):
    """Tool for extracting specific content from a webpage"""
    
    name = "extract_content"
    description = "Extract specific content from a webpage using CSS selectors"
    inputs = {
        "url": {
            "type": "string", 
            "description": "URL of the webpage to extract content from"
        },
        "css_selector": {
            "type": "string", 
            "description": "CSS selector to extract content (e.g., 'div.content', 'h1', 'p.summary')"
        }
    }
    output_type = "string"
    
    def forward(self, url: str, css_selector: str) -> str:
        """Extract content from a webpage using a CSS selector
        
        Args:
            url: URL of the webpage
            css_selector: CSS selector to extract content
            
        Returns:
            Extracted content as text
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Add http:// if missing
            if not url.startswith('http'):
                url = 'https://' + url
            
            # Get the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content using the CSS selector
            elements = soup.select(css_selector)
            
            if not elements:
                return f"No elements found matching selector '{css_selector}'"
            
            # Extract text from all matching elements
            content = "\n".join([el.get_text().strip() for el in elements])
            
            return f"EXTRACTED CONTENT:\n{content}"
            
        except requests.exceptions.RequestException as e:
            return f"Error accessing webpage: {str(e)}"
        except Exception as e:
            return f"Error extracting content: {str(e)}"

class SaveWebContentTool(Tool):
    """Tool for saving web content to a file"""
    
    name = "save_content"
    description = "Save web content to a file for later reference"
    inputs = {
        "filename": {
            "type": "string", 
            "description": "Name of the file to save (will be saved in the 'downloads' folder)"
        },
        "content": {
            "type": "string", 
            "description": "Content to save to the file"
        }
    }
    output_type = "string"
    
    def forward(self, filename: str, content: str) -> str:
        """Save content to a file
        
        Args:
            filename: Name of the file to save
            content: Content to save
            
        Returns:
            Confirmation message
        """
        try:
            # Create downloads directory if it doesn't exist
            os.makedirs('downloads', exist_ok=True)
            
            # Ensure filename has a valid extension
            if not filename.endswith(('.txt', '.html', '.json', '.md', '.csv')):
                filename += '.txt'  # Default to .txt
            
            # Create the full file path
            file_path = os.path.join('downloads', filename)
            
            # Write the content to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Content saved to {file_path}"
            
        except Exception as e:
            return f"Error saving content: {str(e)}"

class WebBrowsingAgent(BaseAgent):
    """Agent for browsing the web, searching for information, and visiting webpages
    Supports multi-agent collaboration for complex web tasks
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
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3-8B-Instruct")
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
                # Create a single agent
                self.main_agent = CodeAgent(
                    tools=tools,
                    model=model,
                    additional_authorized_imports=self.authorized_imports,
                    verbosity_level=1
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
            ExtractWebContentTool(),
            SaveWebContentTool()
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
            description="Runs web searches and visits webpages for you."
        )
        
        # Create a manager agent that handles the overall task (using CodeAgent)
        self.manager_agent = CodeAgent(
            tools=[],  # No direct tools, will use the web_agent
            model=model,
            managed_agents=[self.web_agent],  # This is how we connect the agents
            additional_authorized_imports=self.authorized_imports
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
            if self.use_multi_agent and self.manager_agent:
                # Format the prompt for the multi-agent setup
                prompt = f"""
I need information about the following topic. Please search the web and gather relevant details:

{input_text}

Please provide a comprehensive answer with facts and information from reliable sources.
"""
                # Run the manager agent which will delegate to the web agent as needed
                result = self.manager_agent.run(prompt)
            else:
                # Format prompt for single agent
                prompt = f"""
I need to find information about: {input_text}

DO NOT try to import any packages directly like 'duckduckgo_search'. Instead, use the tools that are already provided:

1. Use web_search(query="your search query") to search the web 
2. Use visit_webpage(url="full URL") to visit specific webpages
3. Use extract_content(url="full URL", css_selector="selector") to extract specific content

Here's an example of how to use these tools correctly:

```python
# Search the web for information
results = web_search(query="Capital of Pakistan")
print("Search results:", results)

# Visit the most relevant page from the results
if results:
    first_result = results[0]
    webpage_content = visit_webpage(url=first_result[0])
    print("Found information:", webpage_content[:500])  # Print first 500 chars
```

Please provide a comprehensive answer based on the information you find.
"""
                # Run the single agent
                result = self.main_agent.run(prompt)
            
            # Add to history
            self.add_to_history(input_text, str(result))
            
            return str(result)
            
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
            "content_saving"
        ]
        
        if self.use_multi_agent:
            capabilities.append("multi_agent_collaboration")
            capabilities.append("task_planning")
        
        return capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation
        
        Returns:
            Dictionary representation of the agent
        """
        agent_dict = super().to_dict()
        agent_dict.update({
            "model_id": self.model_id,
            "use_multi_agent": self.use_multi_agent,
            "capabilities": self.get_capabilities()
        })
        return agent_dict