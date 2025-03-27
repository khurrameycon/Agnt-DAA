"""
Base Agent class for SagaX1
Provides common functionality for all agent types
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import logging

class BaseAgent(ABC):
    """Base class for all agents in SagaX1"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the base agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.history = []
        self.max_history = config.get("max_history", 100)
    
    @abstractmethod
    def run(self, input_text: str, 
            callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input
        
        Args:
            input_text: Input text for the agent
            callback: Optional callback for streaming responses
            
        Returns:
            Agent output text
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's state"""
        pass
    
    def add_to_history(self, user_input: str, agent_output: str) -> None:
        """Add an interaction to the conversation history
        
        Args:
            user_input: User input text
            agent_output: Agent output text
        """
        self.history.append({
            "user_input": user_input,
            "agent_output": agent_output
        })
        
        # Trim history if it exceeds max size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history
        
        Returns:
            List of conversation history entries
        """
        return self.history
    
    def clear_history(self) -> None:
        """Clear the conversation history"""
        self.history = []
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation
        
        Returns:
            Dictionary representation of the agent
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "config": self.config,
            "history_length": len(self.history)
        }