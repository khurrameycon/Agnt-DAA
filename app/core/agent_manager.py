"""
Agent Manager for SagaX1
Manages the creation, configuration, and execution of agents
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable

from app.core.config_manager import ConfigManager
from app.core.model_manager import ModelManager
from app.agents.local_model_agent import LocalModelAgent
from app.agents.web_browsing_agent import WebBrowsingAgent
from app.agents.visual_web_agent import VisualWebAgent
from app.agents.code_gen_agent import CodeGenerationAgent
from app.agents.media_generation_agent import MediaGenerationAgent
from app.agents.agent_registry import AgentRegistry
from smolagents import Tool, DuckDuckGoSearchTool
from app.agents.fine_tuning_agent import FineTuningAgent

class AgentManager:
    """Manages agents and their execution"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the agent manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.active_agents = {}
        self.agent_configs = {}
        
        # Initialize model manager
        self.model_manager = ModelManager(config_manager)
        
        # Initialize available tools
        self.available_tools = self._initialize_available_tools()
        
        # Register agent types
        self._register_agent_types()
        
        # Create default agent if specified in config
        self._create_default_agent()
    
    def _initialize_available_tools(self) -> Dict[str, Tool]:
        """Initialize the available tools for agents
        
        Returns:
            Dictionary of available tools
        """
        tools = {}
        
        # Add web search tool
        web_search_tool = DuckDuckGoSearchTool()
        tools[web_search_tool.name] = web_search_tool
        
        return tools
    
    def _register_agent_types(self) -> None:
        """Register available agent types"""
        AgentRegistry.register("local_model", LocalModelAgent)
        AgentRegistry.register("web_browsing", WebBrowsingAgent)
        AgentRegistry.register("visual_web", VisualWebAgent)
        AgentRegistry.register("code_generation", CodeGenerationAgent)
        AgentRegistry.register("media_generation", MediaGenerationAgent)
        AgentRegistry.register("fine_tuning", FineTuningAgent)
        # AgentRegistry.register("chat", LocalModelAgent)


    def _create_default_agent(self) -> None:
        """Create default agent if specified in config"""
        default_agent = self.config_manager.get("agents.default_agent")
        
        if default_agent:
            # Check if we have a configuration for this agent
            default_config = self.config_manager.get(f"agents.configurations.{default_agent}")
            
            if default_config:
                try:
                    agent_id = f"default_{default_agent}"
                    agent_type = default_config.get("agent_type", "local_model")
                    model_id = default_config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
                    
                    self.create_agent(
                        agent_id=agent_id,
                        agent_type=agent_type,
                        model_config={"model_id": model_id},
                        tools=default_config.get("tools", ["web_search"]),
                        additional_config=default_config.get("additional_config", {})
                    )
                    
                    self.logger.info(f"Created default agent {agent_id}")
                except Exception as e:
                    self.logger.error(f"Error creating default agent: {str(e)}")
    
    def get_available_agent_types(self) -> List[str]:
        """Get the list of available agent types
        
        Returns:
            List of agent type names
        """
        return AgentRegistry.get_registered_types()
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools
        
        Returns:
            List of tool information
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputs": tool.inputs,
                "output_type": tool.output_type
            }
            for tool in self.available_tools.values()
        ]
    
    def create_agent(self, 
                   agent_id: str, 
                   agent_type: str, 
                   model_config: Dict[str, Any],
                   tools: List[str] = None,
                   additional_config: Dict[str, Any] = None) -> str:
        """Create a new agent with the specified configuration
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent to create
            model_config: Model configuration
            tools: List of tool names to include
            additional_config: Additional configuration parameters
            
        Returns:
            ID of the created agent
        """
        if not agent_id:
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"
            
        if agent_id in self.active_agents:
            self.logger.warning(f"Agent with ID {agent_id} already exists. It will be replaced.")
        
        # Save agent configuration for later recreation if needed
        agent_config = {
            "agent_type": agent_type,
            "model_config": model_config,
            "tools": tools or [],
            "additional_config": additional_config or {}
        }
        
        self.agent_configs[agent_id] = agent_config
        
        # Create the agent instance
        try:
            # For specific agent types, handle specialized configuration
            if agent_type == "web_browsing":
                # Create a web browsing agent
                agent_instance = WebBrowsingAgent(
                    agent_id=agent_id,
                    config={
                        **model_config,
                        # Support multi-agent architecture for web browsing
                        "multi_agent": additional_config.get("multi_agent", False),
                        **additional_config
                    }
                )
                self.active_agents[agent_id] = agent_instance
                self.logger.info(f"Created web browsing agent with ID {agent_id}" + 
                                 (" (multi-agent)" if additional_config.get("multi_agent") else ""))
                return agent_id
            else:
                # Use the registry for other agent types
                agent = AgentRegistry.create_agent(agent_id, agent_type, {
                    **model_config,
                    **additional_config
                })
                
                if agent:
                    self.active_agents[agent_id] = agent
                    self.logger.info(f"Created agent with ID {agent_id} of type {agent_type}")
                    return agent_id
                else:
                    self.logger.error(f"Failed to create agent with ID {agent_id}")
                    return None
                
        except Exception as e:
            self.logger.error(f"Error creating agent: {str(e)}")
            return None
    
    def run_agent(self, 
                agent_id: str, 
                input_text: str,
                callback: Optional[Callable[[str], None]] = None) -> str:
        """Run an agent with the given input
        
        Args:
            agent_id: ID of the agent to run
            input_text: Input text for the agent
            callback: Optional callback function for streaming output
            
        Returns:
            Agent output
        """
        if agent_id not in self.agent_configs:
            raise ValueError(f"Agent with ID {agent_id} does not exist")
        
        # Get or create the agent instance
        agent = self.active_agents.get(agent_id)
        
        if agent is None:
            # Try to recreate the agent
            self.create_agent(
                agent_id=agent_id,
                agent_type=self.agent_configs[agent_id]["agent_type"],
                model_config=self.agent_configs[agent_id]["model_config"],
                tools=self.agent_configs[agent_id]["tools"],
                additional_config=self.agent_configs[agent_id]["additional_config"]
            )
            
            agent = self.active_agents.get(agent_id)
            
            if agent is None:
                self.logger.error(f"Failed to create agent with ID {agent_id}")
                return f"Error: Unable to create agent {agent_id}"
        
        # Run the agent
        self.logger.info(f"Running agent {agent_id} with input: {input_text[:50]}...")
        
        try:
            return agent.run(input_text, callback=callback)
        except Exception as e:
            error_msg = f"Error running agent {agent_id}: {str(e)}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get the configuration for an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent configuration dictionary
        """
        if agent_id not in self.agent_configs:
            raise ValueError(f"Agent with ID {agent_id} does not exist")
        
        return self.agent_configs[agent_id]
    
    def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get the list of active agents
        
        Returns:
            List of agent information
        """
        return [
            {
                "agent_id": agent_id,
                "agent_type": self.agent_configs[agent_id]["agent_type"],
                "model_id": self.agent_configs[agent_id]["model_config"].get("model_id", "unknown"),
                "tools": self.agent_configs[agent_id]["tools"],
                "multi_agent": self.agent_configs[agent_id]["additional_config"].get("multi_agent", False) 
                if self.agent_configs[agent_id]["agent_type"] == "web_browsing" else False
            }
            for agent_id in self.active_agents.keys()
        ]
    
    def reset_agent(self, agent_id: str) -> bool:
        """Reset an agent's state
        
        Args:
            agent_id: ID of the agent to reset
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.active_agents:
            self.logger.warning(f"Agent with ID {agent_id} is not active")
            return False
        
        try:
            self.active_agents[agent_id].reset()
            self.logger.info(f"Reset agent {agent_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error resetting agent {agent_id}: {str(e)}")
            return False
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent
        
        Args:
            agent_id: ID of the agent to remove
        """
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
        
        if agent_id in self.agent_configs:
            del self.agent_configs[agent_id]
            
        self.logger.info(f"Removed agent with ID {agent_id}")
    
    def get_agent_history(self, agent_id: str) -> List[Dict[str, str]]:
        """Get the conversation history for an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of conversation entries
        """
        if agent_id not in self.active_agents:
            self.logger.warning(f"Agent with ID {agent_id} is not active")
            return []
        
        try:
            return self.active_agents[agent_id].get_history()
        except Exception as e:
            self.logger.error(f"Error getting history for agent {agent_id}: {str(e)}")
            return []
    
    def clear_agent_history(self, agent_id: str) -> bool:
        """Clear the conversation history for an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.active_agents:
            self.logger.warning(f"Agent with ID {agent_id} is not active")
            return False
        
        try:
            self.active_agents[agent_id].clear_history()
            self.logger.info(f"Cleared history for agent {agent_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing history for agent {agent_id}: {str(e)}")
            return False