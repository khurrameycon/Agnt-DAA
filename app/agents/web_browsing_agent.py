"""
Web Browsing Agent for sagax1
Agent for browsing the web, searching for information, and visiting webpages
Enhanced with direct LLM summarization capability (both local and API modes)
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
    Enhanced with direct LLM summarization capability (both local and API modes)
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
                use_api: Whether to use the Hugging Face Inference API (remote execution)
                use_local_execution: Whether to use local execution (download model)
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
                
        # Get execution mode - prioritize explicit flags
        self.use_api = config.get("use_api", False)
        self.use_local_execution = config.get("use_local_execution", not self.use_api)
        
        # If both flags are somehow set (shouldn't happen), prioritize API mode
        if self.use_api and self.use_local_execution:
            self.logger.warning("Both use_api and use_local_execution are set to True. Prioritizing API mode.")
            self.use_local_execution = False
        
        self.use_multi_agent = config.get("multi_agent", False)
        
        self.main_agent = None
        self.web_agent = None
        self.manager_agent = None
        self.model = None
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
            
            # Initialize the LLM model (local or API-based)
            if self.use_api:
                self._initialize_api_model()
            else:
                # Local execution mode - download model and use locally
                self._initialize_local_model()
            
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
        """Initialize the model by wrapping a direct HTTP call to the HF Inference API."""
        from app.core.config_manager import ConfigManager
        import requests

        self.logger.info("Initializing model via Hugging Face Inference API")

        # 1. Retrieve API key
        api_key = ConfigManager().get_hf_api_key()
        self.logger.info(f"Loaded HF API key: {api_key!r}")
        if not api_key:
            self.logger.error("No API key found. Cannot use Inference API.")
            raise ValueError("HuggingFace API key is required for Inference API mode")
        
        # 2. Prepare headers once
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # 3. Build the base URL for your model
        base_url = f"https://router.huggingface.co/hf-inference/models/{self.model_id}/v1/chat/completions"

        # 4. The wrapper function
        def generate_text(messages):
            try:
                # --- extract prompt from messages (your existing logic) ---
                prompt = ""
                if isinstance(messages, list) and messages:
                    last = messages[-1]
                    if isinstance(last, dict) and "content" in last:
                        content = last["content"]
                        if isinstance(content, list):
                            prompt = " ".join(
                                item.get("text", "")
                                for item in content
                                if item.get("type") == "text"
                            )
                        else:
                            prompt = content
                    else:
                        prompt = str(last)
                # ----------------------------------------------------------

                # 5. Build payload exactly as in your test
                payload = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": float(self.temperature),
                }

                # 6. POST to the inference endpoint
                resp = requests.post(base_url, headers=headers, json=payload)

                # 7. Error handling
                if resp.status_code != 200:
                    self.logger.error(f"Inference API HTTP {resp.status_code}: {resp.text}")
                    return f"Error calling Inference API: {resp.status_code} {resp.text}"

                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                self.logger.error(f"Inference API exception: {e}")
                return f"Error calling Inference API: {e}"

        # 8. Bind it and log
        self.model = generate_text
        self.logger.info(f"Initialized {self.model_id} via direct HTTP inference")

    def _initialize_local_model(self):
        """Initialize the model locally"""
        try:
            from smolagents import TransformersModel
            
            self.logger.info(f"Initializing local model {self.model_id}")
            
            self.model = TransformersModel(
                model_id=self.model_id,
                device_map=self.device,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                trust_remote_code=True,
                do_sample=True  # Add this to fix the temperature warning
            )
            
            self.logger.info(f"Initialized {self.model_id} for local execution")
            
        except Exception as e:
            self.logger.error(f"Error initializing local model: {str(e)}")
            # Try fallbacks like in LocalModelAgent
            self._initialize_with_fallbacks()
    
    def _initialize_with_fallbacks(self):
        """Try alternative model implementations if TransformersModel fails"""
        from smolagents import HfApiModel, OpenAIServerModel, LiteLLMModel
        
        try:
            # Try HfApiModel
            try:
                self.logger.info("Trying HfApiModel...")
                self.model = HfApiModel(
                    model_id=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize with HfApiModel: {str(e)}")
                
                # Try OpenAIServerModel
                try:
                    self.logger.info("Trying OpenAIServerModel...")
                    self.model = OpenAIServerModel(
                        model_id=self.model_id,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to initialize with OpenAIServerModel: {str(e)}")
                    
                    # Try LiteLLMModel as last resort
                    self.logger.info("Trying LiteLLMModel...")
                    self.model = LiteLLMModel(
                        model_id=self.model_id,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
            
            self.is_initialized = True
            self.logger.info(f"Model {self.model_id} initialized successfully with fallback")
            
        except Exception as e:
            self.logger.error(f"All fallback initialization attempts failed: {str(e)}")
            raise
    
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
        
        # Call the model (which could be local or API-based)
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
        
        if self.use_api:
            capabilities.append("api_inference")
        else:
            capabilities.append("local_model_inference")
        
        return capabilities