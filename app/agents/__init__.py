"""
Agent implementations for sagax1
"""

from app.agents.base_agent import BaseAgent
from app.agents.local_model_agent import LocalModelAgent
from app.agents.web_browsing_agent import WebBrowsingAgent
from app.agents.visual_web_agent import VisualWebAgent
from app.agents.code_gen_agent import CodeGenerationAgent
from app.agents.agent_registry import AgentRegistry

__all__ = [
    "BaseAgent",
    "LocalModelAgent", 
    "WebBrowsingAgent", 
    "VisualWebAgent", 
    "CodeGenerationAgent", 
    "AgentRegistry"
]