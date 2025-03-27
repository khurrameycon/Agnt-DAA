"""
Web Browsing Agent for SagaX1
Agent for browsing the web, searching for information, and visiting webpages
Supports multi-agent collaboration for complex web tasks
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
    CodeAgent
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
        self.use_multi_agent = config.get("multi_agent", False)
        
        self.main_agent = None
        self.planner_agent = None
        self.browser_agent = None
        self.summary_agent = None
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

            # If multi-agent mode is enabled, initialize multiple agents
            if self.use_multi_agent:
                self._initialize_multi_agents(model, tools)
            else:
                # Use a single agent
                self.main_agent = CodeAgent(
                    tools=tools,
                    model=model,
                    system_prompt="""You are a web browsing agent that can search the web and visit webpages to find information.
You have access to these tools:
- web_search: Search the web for information
- visit_webpage: Visit a specific webpage and read its content
- extract_content: Extract specific content from a webpage using CSS selectors
- save_content: Save important content to files for future reference

ALWAYS REMEMBER: When visiting a webpage, use the full URL including http:// or https://
For any search or browsing task, first search for relevant information, then visit specific pages if needed.
When extracting content, use specific CSS selectors like 'h1', 'div.content', 'p.summary', etc.""",
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
                trust_remote_code=True
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
    
    def _initialize_multi_agents(self, model, tools: List[Tool]) -> None:
        """Initialize multiple specialized agents
        
        Args:
            model: Language model to use
            tools: List of tools to use
        """
        # Create a planner agent with specific system prompt for planning tasks
        self.planner_agent = CodeAgent(
            tools=[],  # No tools for the planner
            model=model,
            system_prompt="""You are a planning agent that breaks down complex web browsing tasks into simple steps.
Your job is to create a plan to accomplish a task by:
1. Understanding what information is needed 
2. Deciding which websites need to be visited
3. Determining what information needs to be extracted
4. Planning how to process and combine the information

Output your plan as a numbered list of specific steps for a web browsing agent to follow.
Be specific about URLs to visit and what to look for on each page.""",
            additional_authorized_imports=self.authorized_imports,
            verbosity_level=1
        )

        # Create the browser agent for executing the plan
        self.browser_agent = CodeAgent(
            tools=tools,
            model=model,
            system_prompt="""You are a web browsing agent that can search the web and visit webpages to find information.
You can use tools to:
- Search the web for information
- Visit specific webpages and read their content
- Extract specific content from webpages using CSS selectors
- Save important content to files for future reference

ALWAYS REMEMBER: When visiting a webpage, use the full URL including http:// or https://
When extracting content, use specific CSS selectors like 'h1', 'div.content', 'p.summary', etc.
Follow the plan provided to you step by step and report what you find at each step.""",
            additional_authorized_imports=self.authorized_imports,
            verbosity_level=1
        )

        # Create a summary agent for synthesizing the gathered content
        self.summary_agent = CodeAgent(
            tools=[],  # No tools for the summary agent
            model=model,
            system_prompt="""You are a summary agent that synthesizes information gathered from the web.
Your job is to:
1. Review all the information collected by the browser agent
2. Organize the information in a clear, logical structure
3. Create a concise but comprehensive summary
4. Identify any additional information that might be needed

Present your summary in a well-formatted way with appropriate headings and sections.""",
            additional_authorized_imports=self.authorized_imports,
            verbosity_level=1
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
            if self.use_multi_agent:
                # Use multiple agents in sequence
                result = self._run_multi_agent(input_text, callback)
            else:
                # Run the single agent
                enhanced_prompt = f"""
You are a web browsing agent that can search the web and visit webpages to find information.
You have access to these tools:
- web_search: Search the web for information
- visit_webpage: Visit a specific webpage and read its content
- extract_content: Extract specific content from a webpage using CSS selectors
- save_content: Save important content to files for future reference

ALWAYS REMEMBER: When visiting a webpage, use the full URL including http:// or https://
For any search or browsing task, first search for relevant information, then visit specific pages if needed.
When extracting content, use specific CSS selectors like 'h1', 'div.content', 'p.summary', etc.

USER QUERY: {input_text}
"""
                result = self.main_agent.run(enhanced_prompt)
            
            # Add to history
            self.add_to_history(input_text, str(result))
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Error running web browsing agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while browsing the web: {error_msg}"
    
    def _run_multi_agent(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the multi-agent workflow
        
        Args:
            input_text: Input text
            callback: Optional callback
            
        Returns:
            Combined result from all agents
        """
        # 1. First, have the planner create a plan
        planner_prompt = f"""I need help with the following web browsing task:

{input_text}

Please create a detailed plan with specific steps to accomplish this task. 
Break it down into clear, actionable steps that a web browsing agent can follow.
"""
        planner_result = self.planner_agent.run(planner_prompt)
        
        # 2. Next, have the browser agent execute the plan
        browser_prompt = f"""Here's a task I need help with:

{input_text}

A planning agent has created the following plan to accomplish this task:

{planner_result}

Please execute this plan step by step using your web browsing tools. 
For each step, report what you find before moving to the next step.
"""
        browser_result = self.browser_agent.run(browser_prompt)
        
        # 3. Finally, have the summary agent create a summary
        summary_prompt = f"""Here's a web browsing task:

{input_text}

The planning agent created this plan:

{planner_result}

The browser agent executed the plan and found the following information:

{browser_result}

Please create a well-organized summary of all the information found.
Focus on providing a comprehensive answer to the original query.
Use headings and sections to organize the information clearly.
"""
        summary_result = self.summary_agent.run(summary_prompt)
        
        # 4. Combine the results
        combined_result = f"""## Multi-Agent Web Browsing Results

### Original Task
{input_text}

### Planning Phase
{planner_result}

### Information Gathering Phase
{browser_result}

### Summary
{summary_result}
"""
        
        return combined_result
    
    def reset(self) -> None:
        """Reset the agent's state"""
        if self.use_multi_agent:
            # Reset each agent
            if self.planner_agent:
                self.planner_agent.memory.reset()
            if self.browser_agent:
                self.browser_agent.memory.reset()
            if self.summary_agent:
                self.summary_agent.memory.reset()
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