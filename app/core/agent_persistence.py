"""
Agent Persistence Manager for sagax1
Handles saving and loading agent configurations
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import shutil

class AgentPersistenceManager:
    """Manages saving and loading agent configurations to/from disk"""
    
    # In app/core/agent_persistence.py
# Make sure the directories are created properly

    def __init__(self, config_path: str = None):
        """Initialize the persistence manager
        
        Args:
            config_path: Path to config directory, defaults to ~/.sagax1
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up config path
        if config_path is None:
            self.config_path = os.path.join(os.path.expanduser("~"), ".sagax1")
        else:
            self.config_path = os.path.expanduser(config_path)
            
        # Create main config directory if it doesn't exist
        os.makedirs(self.config_path, exist_ok=True)
        
        # Create agents directory within config path
        self.agents_dir = os.path.join(self.config_path, "agents")
        os.makedirs(self.agents_dir, exist_ok=True)
        
        self.logger.info(f"Agent persistence initialized at {self.agents_dir}")
    
    def save_agent_config(self, agent_id: str, config: Dict[str, Any]) -> bool:
        """Save agent configuration to disk
        
        Args:
            agent_id: Agent ID
            config: Agent configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create agent-specific directory
            agent_dir = os.path.join(self.agents_dir, agent_id)
            os.makedirs(agent_dir, exist_ok=True)
            
            # Save config file
            config_file = os.path.join(agent_dir, "config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Saved agent configuration for {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving agent configuration for {agent_id}: {str(e)}")
            return False
    
    def load_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent configuration from disk
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent configuration dictionary or None if not found
        """
        try:
            # Check if agent directory exists
            agent_dir = os.path.join(self.agents_dir, agent_id)
            if not os.path.exists(agent_dir):
                self.logger.warning(f"Agent directory not found for {agent_id}")
                return None
            
            # Load config file
            config_file = os.path.join(agent_dir, "config.json")
            if not os.path.exists(config_file):
                self.logger.warning(f"Config file not found for {agent_id}")
                return None
                
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"Loaded agent configuration for {agent_id}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading agent configuration for {agent_id}: {str(e)}")
            return None
    
    def get_all_agent_ids(self) -> List[str]:
        """Get list of all saved agent IDs
        
        Returns:
            List of agent IDs
        """
        try:
            # List all directories in agents directory
            agent_ids = []
            for item in os.listdir(self.agents_dir):
                item_path = os.path.join(self.agents_dir, item)
                if os.path.isdir(item_path):
                    config_file = os.path.join(item_path, "config.json")
                    if os.path.exists(config_file):
                        agent_ids.append(item)
            
            return agent_ids
            
        except Exception as e:
            self.logger.error(f"Error getting agent IDs: {str(e)}")
            return []
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent configuration from disk
        
        Args:
            agent_id: Agent ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if agent directory exists
            agent_dir = os.path.join(self.agents_dir, agent_id)
            if not os.path.exists(agent_dir):
                self.logger.warning(f"Agent directory not found for {agent_id}")
                return False
            
            # Remove agent directory
            shutil.rmtree(agent_dir)
            
            self.logger.info(f"Deleted agent configuration for {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting agent configuration for {agent_id}: {str(e)}")
            return False
    
    def save_agent_history(self, agent_id: str, history: List[Dict[str, str]]) -> bool:
        """Save agent conversation history to disk
        
        Args:
            agent_id: Agent ID
            history: Conversation history
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create agent-specific directory
            agent_dir = os.path.join(self.agents_dir, agent_id)
            os.makedirs(agent_dir, exist_ok=True)
            
            # Save history file
            history_file = os.path.join(agent_dir, "history.json")
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.info(f"Saved conversation history for {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving conversation history for {agent_id}: {str(e)}")
            return False
    
    def load_agent_history(self, agent_id: str) -> Optional[List[Dict[str, str]]]:
        """Load agent conversation history from disk
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Conversation history or None if not found
        """
        try:
            # Check if agent directory exists
            agent_dir = os.path.join(self.agents_dir, agent_id)
            if not os.path.exists(agent_dir):
                self.logger.warning(f"Agent directory not found for {agent_id}")
                return None
            
            # Load history file
            history_file = os.path.join(agent_dir, "history.json")
            if not os.path.exists(history_file):
                self.logger.warning(f"History file not found for {agent_id}")
                return None
                
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            self.logger.info(f"Loaded conversation history for {agent_id}")
            return history
            
        except Exception as e:
            self.logger.error(f"Error loading conversation history for {agent_id}: {str(e)}")
            return None