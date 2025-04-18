"""
Chat Agent for sagax1
Simple agent for conversational interactions without code execution
"""

import os
import logging
from typing import Dict, Any, List, Optional, Callable

from app.agents.base_agent import BaseAgent

class ChatAgent(BaseAgent):
    """Agent for simple chat conversations"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the chat agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                max_tokens: Maximum number of tokens to generate
                temperature: Temperature for generation
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
        self.device = config.get("device", "auto")
        self.max_new_tokens = config.get("max_tokens", 2048)  # Using max_new_tokens internally
        self.temperature = config.get("temperature", 0.7)
        
        self.model = None
        self.conversation_history = []
        self.is_initialized = False
        
    def initialize(self) -> None:
        """Initialize the model"""
        if self.is_initialized:
            return
        
        try:
            from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
            
            self.logger.info(f"Initializing chat agent with model {self.model_id}")
            
            # Try to create the model based on the model_id
            try:
                # First try TransformersModel for local models
                self.model = TransformersModel(
                    model_id=self.model_id,
                    device_map=self.device,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    trust_remote_code=True,
                    do_sample=True  # Add this to fix temperature warning
                )
                self.logger.info(f"Using TransformersModel for {self.model_id}")
            except Exception as e:
                self.logger.warning(f"Failed to load model with TransformersModel: {str(e)}")
                
                # Try HfApiModel for API-based models
                try:
                    self.model = HfApiModel(
                        model_id=self.model_id,
                        max_tokens=self.max_new_tokens,
                        temperature=self.temperature
                    )
                    self.logger.info(f"Using HfApiModel for {self.model_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to load model with HfApiModel: {str(e)}")
                    
                    # Fallback to OpenAI-compatible API
                    try:
                        self.model = OpenAIServerModel(
                            model_id=self.model_id,
                            max_tokens=self.max_new_tokens,
                            temperature=self.temperature
                        )
                        self.logger.info(f"Using OpenAIServerModel for {self.model_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load model with OpenAIServerModel: {str(e)}")
                        
                        # Final fallback to LiteLLM
                        self.model = LiteLLMModel(
                            model_id=self.model_id,
                            max_tokens=self.max_new_tokens,
                            temperature=self.temperature
                        )
                        self.logger.info(f"Using LiteLLMModel for {self.model_id}")
            
            self.is_initialized = True
            self.logger.info(f"Chat agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing chat agent: {str(e)}")
            raise
    
    def _format_conversation_for_context(self, max_tokens: int = 1000) -> str:
        """Format conversation history for context
        
        Args:
            max_tokens: Maximum approximate tokens to include
            
        Returns:
            Formatted conversation history
        """
        if not self.conversation_history:
            return ""
        
        # Start with most recent history (limited by token count)
        formatted_history = []
        estimated_tokens = 0
        
        for entry in reversed(self.conversation_history):
            # Rough token estimation (approx 4 chars per token)
            user_tokens = len(entry["user_input"]) // 4
            agent_tokens = len(entry["agent_output"]) // 4
            entry_tokens = user_tokens + agent_tokens + 10  # +10 for formatting
            
            if estimated_tokens + entry_tokens > max_tokens:
                break
                
            formatted_history.append(f"User: {entry['user_input']}\nAssistant: {entry['agent_output']}")
            estimated_tokens += entry_tokens
        
        # Reverse back to chronological order
        formatted_history.reverse()
        return "\n\n".join(formatted_history)
    
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
            # Format with conversation history for context
            conversation_context = self._format_conversation_for_context()
            
            if conversation_context:
                prompt = f"{conversation_context}\n\nUser: {input_text}\nAssistant:"
            else:
                prompt = f"User: {input_text}\nAssistant:"
            
            # Direct model generation without code execution
            response = self.model.generate(prompt)
            
            # Add to history
            self.add_to_history(input_text, response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error: {error_msg}"
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        return ["conversation", "text_generation"]