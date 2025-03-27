"""
Agent Registry for SagaX1
Manages registration and retrieval of agent types
"""

from typing import Dict, Any, Type, List, Optional, Callable
import logging
from app.agents.base_agent import BaseAgent

class AgentRegistry:
    """Registry for agent types"""
    
    _registry = {}
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def register(cls, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent type
        
        Args:
            agent_type: Name of the agent type
            agent_class: Agent class
        """
        if agent_type in cls._registry:
            cls._logger.warning(f"Agent type '{agent_type}' already registered. It will be overwritten.")
        
        cls._registry[agent_type] = agent_class
        cls._logger.info(f"Registered agent type: {agent_type}")
    
    @classmethod
    def get_agent_class(cls, agent_type: str) -> Optional[Type[BaseAgent]]:
        """Get the agent class for a given type
        
        Args:
            agent_type: Name of the agent type
            
        Returns:
            Agent class or None if not found
        """
        return cls._registry.get(agent_type)
    
    @classmethod
    def create_agent(cls, 
                   agent_id: str, 
                   agent_type: str, 
                   config: Dict[str, Any]) -> Optional[BaseAgent]:
        """Create an agent instance of the specified type
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent to create
            config: Agent configuration
            
        Returns:
            Instance of the agent or None if the type is not registered
        """
        agent_class = cls.get_agent_class(agent_type)
        
        if agent_class is None:
            cls._logger.error(f"Agent type '{agent_type}' not found in registry")
            return None
        
        try:
            return agent_class(agent_id, config)
        except Exception as e:
            cls._logger.exception(f"Error creating agent of type '{agent_type}': {str(e)}")
            return None
    
    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get the list of registered agent types
        
        Returns:
            List of registered agent type names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def unregister(cls, agent_type: str) -> bool:
        """Unregister an agent type
        
        Args:
            agent_type: Name of the agent type to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if agent_type in cls._registry:
            del cls._registry[agent_type]
            cls._logger.info(f"Unregistered agent type: {agent_type}")
            return True
        
        return False