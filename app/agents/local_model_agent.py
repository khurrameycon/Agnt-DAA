"""
Local Model Agent for SagaX1
Runs local Hugging Face models for text generation and task completion
"""

import os
import logging
from typing import Dict, Any, List, Optional, Callable

from app.agents.base_agent import BaseAgent
from smolagents import TransformersModel, CodeAgent, DuckDuckGoSearchTool
from huggingface_hub import snapshot_download

class LocalModelAgent(BaseAgent):
    """Agent for running local models from Hugging Face"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the local model agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                agent_type: Type of agent to use ('code_agent' or 'tool_calling_agent')
                tools: List of tools to use
                authorized_imports: List of authorized imports for code agent
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3-8B-Instruct")
        self.device = config.get("device", "auto")
        self.agent_type = config.get("agent_type", "code_agent")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        self.tools_config = config.get("tools", [])
        self.authorized_imports = config.get("authorized_imports", [])
        
        self.model = None
        self.agent = None
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the model and agent"""
        if self.is_initialized:
            return
            
        # Download the model if needed
        self._ensure_model_downloaded()
        
        # Create the model
        self.logger.info(f"Initializing model {self.model_id}")
        
        try:
            self.model = TransformersModel(
                model_id=self.model_id,
                device_map=self.device,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                trust_remote_code=True
            )
            
            # Create tools 
            tools = self._initialize_tools()
            
            # Create the agent based on agent_type
            if self.agent_type == "code_agent":
                self.agent = CodeAgent(
                    tools=tools,
                    model=self.model,
                    additional_authorized_imports=self.authorized_imports,
                    verbosity_level=1
                )
            else:
                # For now, we'll just use CodeAgent as default
                # In a later phase, we'll add ToolCallingAgent
                self.agent = CodeAgent(
                    tools=tools,
                    model=self.model,
                    additional_authorized_imports=self.authorized_imports,
                    verbosity_level=1
                )
            
            self.is_initialized = True
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing agent: {str(e)}")
            raise
    
    def _ensure_model_downloaded(self) -> None:
        """Download the model if needed"""
        # We'll use huggingface_hub to download the model
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            
            # Try to download model card first as a test
            hf_hub_download(
                repo_id=self.model_id,
                filename="config.json",
                token=os.environ.get("HF_API_KEY")
            )
            
            # If successful, model is available
            self.logger.info(f"Model {self.model_id} is available")
            
        except Exception as e:
            self.logger.error(f"Error checking model availability: {str(e)}")
            raise
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize tools for the agent"""
        tools = []
        
        # For now, we'll just add the web search tool
        # In later phases, we'll dynamically create tools based on config
        tools.append(DuckDuckGoSearchTool())
        
        return tools
    
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
            # Run the agent
            result = self.agent.run(input_text)
            
            # Add to history
            self.add_to_history(input_text, str(result))
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Error running agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error: {error_msg}"
    
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
        return ["text_generation", "code_execution", "web_search"]