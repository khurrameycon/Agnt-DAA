"""
Visual Web Automation Tab Component for SagaX1
Tab for visual web automation agent interaction
Enhanced with features from the web_browser.ipynb example
"""

import os
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QScrollArea, 
    QMessageBox, QGroupBox, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage, QFont
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
        
        # Add Chrome requirement notice
        self.add_chrome_notice()
    
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
        self.screenshot_label.setMinimumHeight(600)  # Increased height for better visibility
        
        # Add to scroll area
        scroll_area.setWidget(self.screenshot_label)
        browser_layout.addWidget(scroll_area)
        
        # Add quick command buttons
        self.add_quick_commands(browser_layout)
        
        # Add browser widget to splitter
        splitter.addWidget(browser_widget)
        
        # Conversation widget
        self.conversation = ConversationWidget()
        splitter.addWidget(self.conversation)
        
        # Set initial sizes for the splitter sections
        splitter.setSizes([600, 300])  # Browser view gets more space
        
        # Add splitter to layout
        self.layout.addWidget(splitter, stretch=1)
    
    def add_quick_commands(self, parent_layout):
        """Add quick command buttons for common actions
        
        Args:
            parent_layout: Parent layout to add the buttons to
        """
        commands_group = QGroupBox("Quick Commands")
        grid_layout = QGridLayout(commands_group)
        
        # Navigation commands
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.execute_quick_command("go_back", ""))
        grid_layout.addWidget(back_button, 0, 0)
        
        forward_button = QPushButton("Forward")
        forward_button.clicked.connect(lambda: self.execute_quick_command("forward", ""))
        grid_layout.addWidget(forward_button, 0, 1)
        
        # Scrolling commands
        scroll_down_button = QPushButton("Scroll Down")
        scroll_down_button.clicked.connect(lambda: self.execute_quick_command("scroll", "500"))
        grid_layout.addWidget(scroll_down_button, 0, 2)
        
        scroll_up_button = QPushButton("Scroll Up")
        scroll_up_button.clicked.connect(lambda: self.execute_quick_command("scroll_up", "500"))
        grid_layout.addWidget(scroll_up_button, 0, 3)
        
        # Other commands
        close_popups_button = QPushButton("Close Popups")
        close_popups_button.clicked.connect(lambda: self.execute_quick_command("close_popups", ""))
        grid_layout.addWidget(close_popups_button, 1, 0)
        
        screenshot_button = QPushButton("Take Screenshot")
        screenshot_button.clicked.connect(lambda: self.execute_quick_command("screenshot", ""))
        grid_layout.addWidget(screenshot_button, 1, 1)
        
        # URL navigation
        url_layout = QHBoxLayout()
        self.url_input = QTextEdit()
        self.url_input.setPlaceholderText("Enter URL...")
        self.url_input.setMaximumHeight(30)  # Make it a single line
        url_layout.addWidget(self.url_input)
        
        go_button = QPushButton("Go")
        go_button.clicked.connect(self.navigate_to_url)
        url_layout.addWidget(go_button)
        
        grid_layout.addLayout(url_layout, 2, 0, 1, 4)  # Span across all columns
        
        parent_layout.addWidget(commands_group)
    
    def navigate_to_url(self):
        """Navigate to the URL entered in the URL input field"""
        if self.current_agent is None:
            self.show_error_message("Browser not started", "Please start the browser first.")
            return
        
        url = self.url_input.toPlainText().strip()
        if not url:
            return
        
        # Execute the go_to command
        self.execute_quick_command("go_to", url)
        
        # Clear the URL input
        self.url_input.clear()
    
    def execute_quick_command(self, command: str, parameters: str):
        """Execute a quick command
        
        Args:
            command: Command to execute
            parameters: Parameters for the command
        """
        if self.current_agent is None:
            self.show_error_message("Browser not started", "Please start the browser first.")
            return
        
        try:
            # Get the visual tool from the agent
            visual_tool = self.current_agent.visual_tool
            
            # Execute the command
            result = visual_tool.forward(command, parameters)
            
            # Add to conversation
            self.conversation.add_message(f"Command: {command} {parameters}", is_user=True)
            self.conversation.add_message(result, is_user=False)
        except Exception as e:
            self.logger.error(f"Error executing quick command: {str(e)}")
            self.conversation.add_message(f"Error executing command: {str(e)}", is_user=False)
    
    def add_chrome_notice(self):
        """Add a notice about Chrome being required"""
        notice = QLabel(
            "<b>Note:</b> Google Chrome is required for Visual Web Automation. "
            "Please make sure Chrome is installed on your system."
        )
        notice.setStyleSheet("color: #FF6700; background-color: #FFEFDB; padding: 8px; border-radius: 4px;")
        notice.setWordWrap(True)
        self.layout.addWidget(notice)
    
    def create_input_panel(self):
        """Create input panel for user commands"""
        input_layout = QVBoxLayout()
        
        # Command type label
        input_layout.addWidget(QLabel("<b>Instructions for the Visual Web Agent:</b>"))
        
        # Command input
        self.command_input = QTextEdit()
        self.command_input.setPlaceholderText("Describe what you want the agent to do in the browser...\nExample: 'Go to wikipedia.org and search for artificial intelligence'")
        self.command_input.setMinimumHeight(100)
        input_layout.addWidget(self.command_input)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Examples button
        examples_button = QPushButton("Show Examples")
        examples_button.clicked.connect(self.show_examples)
        buttons_layout.addWidget(examples_button)
        
        # Clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_input)
        buttons_layout.addWidget(clear_button)
        
        buttons_layout.addStretch()
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px;")
        self.send_button.clicked.connect(self.send_command)
        self.send_button.setEnabled(False)
        buttons_layout.addWidget(self.send_button)
        
        input_layout.addLayout(buttons_layout)
        
        self.layout.addLayout(input_layout)
    
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
    
    def clear_input(self):
        """Clear the command input field"""
        self.command_input.clear()
    
    def show_examples(self):
        """Show examples of visual web automation commands"""
        examples = [
            "Go to wikipedia.org and search for artificial intelligence",
            "Visit github.com/trending and tell me the top trending repository today",
            "Go to amazon.com and find the best-selling book in science fiction",
            "Navigate to news.ycombinator.com and summarize the top 3 stories",
            "Browse to weather.com and tell me the forecast for New York City"
        ]
        
        example_message = "Example tasks for the Visual Web Agent:\n\n"
        for i, example in enumerate(examples, 1):
            example_message += f"{i}. {example}\n"
            
        QMessageBox.information(
            self,
            "Visual Web Automation Examples",
            example_message
        )
    
    def show_error_message(self, title: str, message: str):
        """Show an error message dialog
        
        Args:
            title: Dialog title
            message: Error message
        """
        QMessageBox.warning(self, title, message)
    
    def toggle_browser(self):
        """Toggle browser start/stop"""
        if self.current_agent_id is None:
            return
        
        # Check if the button text is "Stop Browser"
        if self.start_stop_button.text() == "Stop Browser":
            # Close the browser
            if self.current_agent:
                self.current_agent.cleanup()
                self.start_stop_button.setText("Start Browser")
                self.send_button.setEnabled(False)
                self.screenshot_label.setPixmap(QPixmap())  # Clear screenshot
                self.screenshot_label.setText("Browser closed")
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
                self.show_error_message("Error", f"Failed to create agent {self.current_agent_id}")
                return
            
            # Connect signals
            agent.screenshot_updated.connect(self.update_screenshot)
            
            # Update current agent
            self.current_agent = agent
        
        try:
            # Show loading message
            self.screenshot_label.setText("Starting browser, please wait...")
            self.repaint()  # Force UI update
            
            # Initialize agent (which starts the browser)
            agent.initialize()
            
            # Update button text
            if hasattr(agent, 'visual_tool') and agent.visual_tool.browser is not None:
                self.start_stop_button.setText("Stop Browser")
                self.send_button.setEnabled(True)
                self.conversation.add_message("Browser started successfully", is_user=False)
            else:
                self.start_stop_button.setText("Start Browser")
                self.send_button.setEnabled(False)
                self.screenshot_label.setText("Failed to start browser")
        except Exception as e:
            self.logger.error(f"Error starting browser: {str(e)}")
            self.start_stop_button.setText("Start Browser")
            self.send_button.setEnabled(False)
            self.screenshot_label.setText(f"Error starting browser: {str(e)}")
            self.show_error_message("Browser Error", f"Could not start Chrome browser: {str(e)}\n\nPlease make sure Google Chrome is installed on your system.")
    
    def send_command(self):
        """Send command to agent"""
        if self.current_agent is None:
            self.show_error_message("Browser not started", "Please start the browser first.")
            return
        
        # Get command
        command = self.command_input.toPlainText().strip()
        if not command:
            self.show_error_message("Empty Command", "Please enter a command for the agent.")
            return
        
        # Add to conversation
        self.conversation.add_message(command, is_user=True)
        
        # Clear input
        self.command_input.clear()
        
        # Disable input
        self.command_input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.start_stop_button.setEnabled(False)
        
        # Show loading message
        self.conversation.add_message("Processing your request...", is_user=False)
        
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
        self.start_stop_button.setEnabled(True)
    
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
                
                # Scale pixmap to fit label while preserving aspect ratio
                pixmap = pixmap.scaled(
                    self.screenshot_label.width(), 
                    self.screenshot_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Set pixmap to label
                self.screenshot_label.setPixmap(pixmap)
                
                # Update current URL in conversation if available
                if self.current_agent and hasattr(self.current_agent, 'visual_tool') and self.current_agent.visual_tool.webdriver:
                    try:
                        current_url = self.current_agent.visual_tool.webdriver.current_url
                        title = self.current_agent.visual_tool.webdriver.title
                        status_text = f"Browser at: {title} ({current_url})"
                        if hasattr(self.parent(), 'status_bar'):
                            self.parent().status_bar.showMessage(status_text, 5000)
                    except:
                        pass
        except Exception as e:
            self.logger.error(f"Error updating screenshot: {str(e)}")
            
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
    
    def create_new_agent(self):
        """Create a new visual web agent"""
        # Find the main window by traversing up the parent hierarchy
        main_window = self
        while main_window and not hasattr(main_window, 'create_visual_web_agent'):
            main_window = main_window.parent()
        
        if main_window and hasattr(main_window, 'create_visual_web_agent'):
            main_window.create_visual_web_agent()
        else:
            # Fallback if we can't find the method
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Not Implemented",
                "Create visual web agent functionality not found in main window."
            )