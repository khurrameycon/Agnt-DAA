"""
Visual Web Automation Tab Component for SagaX1
Tab for visual web automation agent interaction
"""

import os
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from app.agents.visual_web_agent import VisualWebAgent
from app.core.agent_manager import AgentManager
from app.ui.components.conversation import ConversationWidget


class VisualWebTab(QWidget):
    """Tab for visual web automation"""
    
    def __init__(self, agent_manager: AgentManager, parent=None):
        """Initialize the visual web tab
        
        Args:
            agent_manager: Agent manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.agent_manager = agent_manager
        self.logger = logging.getLogger(__name__)
        self.current_agent_id = None
        self.current_agent = None
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create top bar
        self.create_top_bar()
        
        # Create main panel
        self.create_main_panel()
        
        # Create input panel
        self.create_input_panel()
    
    def create_top_bar(self):
        """Create top bar with agent selection and controls"""
        top_layout = QHBoxLayout()
        
        # Agent selection
        top_layout.addWidget(QLabel("Select Visual Web Agent:"))
        self.agent_selector = QComboBox()
        self.agent_selector.currentTextChanged.connect(self.on_agent_selected)
        top_layout.addWidget(self.agent_selector)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_agents)
        top_layout.addWidget(refresh_button)
        
        # Create button
        create_button = QPushButton("Create New")
        create_button.clicked.connect(self.create_new_agent)
        top_layout.addWidget(create_button)
        
        # Start/stop button
        self.start_stop_button = QPushButton("Start Browser")
        self.start_stop_button.clicked.connect(self.toggle_browser)
        self.start_stop_button.setEnabled(False)
        top_layout.addWidget(self.start_stop_button)
        
        top_layout.addStretch()
        
        self.layout.addLayout(top_layout)
    
    def create_main_panel(self):
        """Create main panel with browser view and conversation history"""
        # Create splitter for browser view and conversation
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Browser view
        browser_widget = QWidget()
        browser_layout = QVBoxLayout(browser_widget)
        
        # Create scroll area for browser view
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create label for browser screenshot
        self.screenshot_label = QLabel("No screenshot available")
        self.screenshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screenshot_label.setMinimumHeight(300)
        
        # Add to scroll area
        scroll_area.setWidget(self.screenshot_label)
        browser_layout.addWidget(scroll_area)
        
        # Add browser widget to splitter
        splitter.addWidget(browser_widget)
        
        # Conversation widget
        self.conversation = ConversationWidget()
        splitter.addWidget(self.conversation)
        
        # Add splitter to layout
        self.layout.addWidget(splitter, stretch=1)
    
    def create_input_panel(self):
        """Create input panel for user commands"""
        input_layout = QHBoxLayout()
        
        # Command input
        self.command_input = QTextEdit()
        self.command_input.setPlaceholderText("Describe what you want the agent to do in the browser...")
        self.command_input.setMaximumHeight(100)
        input_layout.addWidget(self.command_input)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_command)
        self.send_button.setEnabled(False)
        input_layout.addWidget(self.send_button)
        
        self.layout.addLayout(input_layout)
    
    def load_agents(self):
        """Load available agents"""
        # Clear existing items
        self.agent_selector.clear()
        
        # Get active agents
        active_agents = self.agent_manager.get_active_agents()
        
        # Filter to visual web agents
        visual_web_agents = [
            agent for agent in active_agents
            if agent["agent_type"] == "visual_web"
        ]
        
        if not visual_web_agents:
            self.agent_selector.addItem("No visual web agents available")
            self.start_stop_button.setEnabled(False)
            self.send_button.setEnabled(False)
            return
        
        # Add to combo box
        for agent in visual_web_agents:
            self.agent_selector.addItem(agent["agent_id"])
        
        # Select first agent
        if self.agent_selector.count() > 0:
            self.agent_selector.setCurrentIndex(0)
    
    def on_agent_selected(self, agent_id: str):
        """Handle agent selection
        
        Args:
            agent_id: ID of the selected agent
        """
        if agent_id == "No visual web agents available":
            self.current_agent_id = None
            self.current_agent = None
            self.start_stop_button.setEnabled(False)
            self.send_button.setEnabled(False)
            return
        
        try:
            # Get agent instance
            agent = self.agent_manager.active_agents.get(agent_id)
            
            # If agent exists, connect signals
            if agent is not None and isinstance(agent, VisualWebAgent):
                # Disconnect from previous agent if any
                if self.current_agent is not None:
                    try:
                        self.current_agent.screenshot_updated.disconnect(self.update_screenshot)
                    except:
                        pass
                
                # Connect to new agent
                agent.screenshot_updated.connect(self.update_screenshot)
                
                # Update current agent
                self.current_agent_id = agent_id
                self.current_agent = agent
                
                # Enable buttons
                self.start_stop_button.setEnabled(True)
                self.send_button.setEnabled(True)
                
                # Update button text
                if hasattr(agent, 'visual_tool') and agent.visual_tool.browser is not None:
                    self.start_stop_button.setText("Stop Browser")
                else:
                    self.start_stop_button.setText("Start Browser")
            else:
                # Create agent if it doesn't exist
                agent_config = self.agent_manager.get_agent_config(agent_id)
                
                if agent_config["agent_type"] == "visual_web":
                    self.current_agent_id = agent_id
                    self.current_agent = None
                    
                    # Enable start button
                    self.start_stop_button.setEnabled(True)
                    self.start_stop_button.setText("Start Browser")
                    
                    # Disable send button until browser is started
                    self.send_button.setEnabled(False)
        except Exception as e:
            self.logger.error(f"Error selecting agent: {str(e)}")
            self.current_agent_id = None
            self.current_agent = None
            self.start_stop_button.setEnabled(False)
            self.send_button.setEnabled(False)
    
    def create_new_agent(self):
        """Create a new visual web agent"""
        # Signal to main window to open the create agent dialog with visual_web type
        self.parent().create_visual_web_agent()
    
    def toggle_browser(self):
        """Toggle browser start/stop"""
        if self.current_agent_id is None:
            return
        
        # Get agent instance
        agent = self.agent_manager.active_agents.get(self.current_agent_id)
        
        if agent is None:
            # Create agent if it doesn't exist
            self.agent_manager.create_agent(
                agent_id=self.current_agent_id,
                agent_type="visual_web",
                model_config=self.agent_manager.get_agent_config(self.current_agent_id)["model_config"],
                tools=self.agent_manager.get_agent_config(self.current_agent_id)["tools"],
                additional_config=self.agent_manager.get_agent_config(self.current_agent_id)["additional_config"]
            )
            
            agent = self.agent_manager.active_agents.get(self.current_agent_id)
            
            if agent is None:
                self.logger.error(f"Failed to create agent {self.current_agent_id}")
                return
            
            # Connect signals
            agent.screenshot_updated.connect(self.update_screenshot)
            
            # Update current agent
            self.current_agent = agent
        
        # Initialize agent (which starts the browser)
        agent.initialize()
        
        # Update button text
        if hasattr(agent, 'visual_tool') and agent.visual_tool.browser is not None:
            self.start_stop_button.setText("Stop Browser")
            self.send_button.setEnabled(True)
        else:
            self.start_stop_button.setText("Start Browser")
            self.send_button.setEnabled(False)
    
    def send_command(self):
        """Send command to agent"""
        if self.current_agent is None:
            return
        
        # Get command
        command = self.command_input.toPlainText().strip()
        if not command:
            return
        
        # Add to conversation
        self.conversation.add_message(command, is_user=True)
        
        # Clear input
        self.command_input.clear()
        
        # Disable input
        self.command_input.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Run agent in thread
        self.parent().run_agent_in_thread(
            self.current_agent_id, 
            command,
            self.handle_agent_result
        )
    
    def handle_agent_result(self, result: str):
        """Handle agent result
        
        Args:
            result: Agent result
        """
        # Add to conversation
        self.conversation.add_message(result, is_user=False)
        
        # Enable input
        self.command_input.setEnabled(True)
        self.send_button.setEnabled(True)
    
    @pyqtSlot(object)
    def update_screenshot(self, screenshot):
        """Update screenshot
        
        Args:
            screenshot: Screenshot image
        """
        try:
            # Convert PIL image to QImage
            if screenshot is not None:
                # Convert PIL Image to QImage
                img = screenshot.convert("RGB")
                width, height = img.size
                bytes_per_line = 3 * width
                q_img = QImage(img.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Create pixmap from QImage
                pixmap = QPixmap.fromImage(q_img)
                
                # Scale pixmap to fit label
                pixmap = pixmap.scaled(
                    self.screenshot_label.width(), 
                    self.screenshot_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Set pixmap to label
                self.screenshot_label.setPixmap(pixmap)
        except Exception as e:
            self.logger.error(f"Error updating screenshot: {str(e)}")