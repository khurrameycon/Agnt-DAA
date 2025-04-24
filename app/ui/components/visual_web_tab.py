"""
Enhanced Visual Web Tab Component for sagax1
Tab for visual web automation agent interaction with VLM capabilities
"""

import os
import logging
from typing import Optional
import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QScrollArea, 
    QMessageBox, QGroupBox, QGridLayout, QCheckBox, QFrame
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from app.core.agent_manager import AgentManager
from app.ui.components.conversation import ConversationWidget


class VisualWebTab(QWidget):
    """Tab for enhanced visual web automation with VLM integration"""
    
    def __init__(self, agent_manager: AgentManager, parent=None):
        """Initialize the enhanced visual web tab
        
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
        refresh_button.clicked.connect(self.refresh_agents)
        top_layout.addWidget(refresh_button)
        
        # Create button
        create_button = QPushButton("Create New")
        create_button.clicked.connect(self.create_agent)
        top_layout.addWidget(create_button)
        
        # Start/stop button
        self.start_stop_button = QPushButton("Start Browser")
        self.start_stop_button.clicked.connect(self.start_stop_browser)
        self.start_stop_button.setEnabled(False)
        top_layout.addWidget(self.start_stop_button)
        
        # Task selector combobox
        self.task_selector = QComboBox()
        self.task_selector.addItems([
            "Suggest Actions",
            "Explain Page Content",
            "Find Elements",
            "Guide Navigation",
            "Custom Analysis"
        ])
        top_layout.addWidget(self.task_selector)
        
        # Analyze button
        self.analyze_button = QPushButton("Analyze Page")
        self.analyze_button.clicked.connect(self.analyze_current_page)
        self.analyze_button.setEnabled(False)
        top_layout.addWidget(self.analyze_button)
        
        top_layout.addStretch()
        
        self.layout.addLayout(top_layout)
    
    def create_main_panel(self):
        """Create main panel with browser view, analysis results, and conversation history"""
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Browser view and analysis panel
        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)
        
        # Left side: Browser view
        browser_panel = QWidget()
        browser_layout = QVBoxLayout(browser_panel)
        
        # Create scroll area for browser view
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create label for browser screenshot
        self.screenshot_label = QLabel("No screenshot available")
        self.screenshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screenshot_label.setMinimumHeight(400)
        
        # Add to scroll area
        scroll_area.setWidget(self.screenshot_label)
        browser_layout.addWidget(scroll_area)
        
        # Add browser controls
        self.add_quick_commands(browser_layout)
        
        top_layout.addWidget(browser_panel, 2)  # Give more space to browser
        
        # Right side: Analysis results
        analysis_panel = QWidget()
        analysis_layout = QVBoxLayout(analysis_panel)
        
        # Analysis text area
        analysis_layout.addWidget(QLabel("Analysis Results:"))
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlaceholderText("Use the 'Analyze Page' button to get insights about the current page...")
        analysis_layout.addWidget(self.analysis_text)
        
        # Suggestions area
        suggestions_group = QGroupBox("Suggested Actions")
        self.suggestions_layout = QVBoxLayout(suggestions_group)
        analysis_layout.addWidget(suggestions_group)
        
        top_layout.addWidget(analysis_panel, 1)
        
        top_panel.setLayout(top_layout)
        splitter.addWidget(top_panel)
        
        # Bottom panel: Conversation
        self.conversation = ConversationWidget()
        splitter.addWidget(self.conversation)
        
        # Set initial sizes
        splitter.setSizes([600, 300])
        
        # Add to main layout
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
    
    def add_chrome_notice(self):
        """Add a notice about Chrome being required"""
        notice = QLabel(
            "<b>Note:</b> Google Chrome is required for Visual Web Automation. "
            "The AI Vision features require a valid Hugging Face API key."
        )
        notice.setStyleSheet("color: #FF6700; background-color: #FFEFDB; padding: 8px; border-radius: 4px;")
        notice.setWordWrap(True)
        self.layout.addWidget(notice)
    
    def on_agent_selected(self, agent_id: str):
        """Handle agent selection
        
        Args:
            agent_id: ID of the selected agent
        """
        self.logger.info(f"Agent selected: '{agent_id}'")
        
        if agent_id == "No visual web agents available":
            self.current_agent_id = None
            self.current_agent = None
            self.start_stop_button.setEnabled(False)
            self.send_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            return
        
        try:
            # Store the selected agent ID immediately to avoid empty ID issues
            self.current_agent_id = agent_id
            
            # Get agent instance
            agent = self.agent_manager.active_agents.get(agent_id)
            
            # If agent exists, connect signals
            if agent is not None and hasattr(agent, 'screenshot_updated'):
                # Disconnect from previous agent if any
                if self.current_agent is not None and hasattr(self.current_agent, 'screenshot_updated'):
                    try:
                        self.current_agent.screenshot_updated.disconnect(self.update_screenshot)
                    except:
                        pass
                    
                # Connect to new agent
                agent.screenshot_updated.connect(self.update_screenshot)
                
                # Update current agent
                self.current_agent = agent
                
                # Enable buttons
                self.start_stop_button.setEnabled(True)
                
                # Enable send button only if browser is started
                browser_started = hasattr(agent, 'browser') and agent.browser is not None
                self.send_button.setEnabled(browser_started)
                self.analyze_button.setEnabled(browser_started)
                
                # Update button text
                if browser_started:
                    self.start_stop_button.setText("Stop Browser")
                else:
                    self.start_stop_button.setText("Start Browser")
            else:
                # Store ID but not the agent instance yet
                self.current_agent = None
                
                # Enable start button
                self.start_stop_button.setEnabled(True)
                self.start_stop_button.setText("Start Browser")
                
                # Disable other buttons until browser is started
                self.send_button.setEnabled(False)
                self.analyze_button.setEnabled(False)
                
        except Exception as e:
            self.logger.error(f"Error selecting agent: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.current_agent = None
            self.start_stop_button.setEnabled(False)
            self.send_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
    
    def refresh_agents(self):
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
            self.analyze_button.setEnabled(False)
            return
        
        # Add to combo box
        for agent in visual_web_agents:
            self.agent_selector.addItem(agent["agent_id"])
        
        # Select first agent
        if self.agent_selector.count() > 0:
            self.agent_selector.setCurrentIndex(0)
    
    def create_agent(self):
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
    
    def start_stop_browser(self):
        """Toggle browser start/stop"""
        if self.current_agent_id is None:
            self.show_error_message("No Agent Selected", "Please select or create an agent first.")
            return
            
        self.logger.info(f"Toggle browser called. Current agent ID: {self.current_agent_id}")
        
        # Check if the button text is "Stop Browser"
        if self.start_stop_button.text() == "Stop Browser":
            # Close the browser
            if self.current_agent:
                self.current_agent.cleanup()
                self.start_stop_button.setText("Start Browser")
                self.send_button.setEnabled(False)
                self.analyze_button.setEnabled(False)
                self.screenshot_label.setPixmap(QPixmap())  # Clear screenshot
                self.screenshot_label.setText("Browser closed")
            return
        
        # Show loading message
        self.screenshot_label.setText("Starting browser, please wait...")
        self.repaint()  # Force UI update
        
        try:
            # Get or create agent instance
            agent = self.agent_manager.active_agents.get(self.current_agent_id)
            
            if agent is None:
                # Create agent if it doesn't exist
                self.logger.info(f"Creating agent: {self.current_agent_id}")
                self.agent_manager.create_agent(
                    agent_id=self.current_agent_id,
                    agent_type="visual_web",
                    model_config=self.agent_manager.get_agent_config(self.current_agent_id)["model_config"],
                    tools=self.agent_manager.get_agent_config(self.current_agent_id)["tools"],
                    additional_config=self.agent_manager.get_agent_config(self.current_agent_id)["additional_config"]
                )
                
                agent = self.agent_manager.active_agents.get(self.current_agent_id)
                
            # Try to disconnect existing signals to avoid duplicates
            try:
                if hasattr(agent, 'screenshot_updated'):
                    agent.screenshot_updated.disconnect(self.update_screenshot)
            except:
                pass
                
            # Connect signals
            if hasattr(agent, 'screenshot_updated'):
                agent.screenshot_updated.connect(self.update_screenshot)
            
            # Update current agent
            self.current_agent = agent
            
            # Initialize the agent (starts the browser)
            self.logger.info("Initializing agent with standard method...")
            agent.initialize()
            
            # Update UI based on browser status
            if hasattr(agent, 'browser') and agent.browser is not None:
                self.logger.info("Browser started with standard method")
                self.start_stop_button.setText("Stop Browser")
                self.send_button.setEnabled(True)
                self.analyze_button.setEnabled(True)
                self.conversation.add_message("Browser started successfully", is_user=False)
                return
            
            # If we get here, there was a problem starting the browser
            self.start_stop_button.setText("Start Browser")
            self.send_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self.screenshot_label.setText("Failed to start browser")
            self.show_error_message("Browser Error", "Failed to start browser. Please check if Chrome is installed and try again.")
                
        except Exception as e:
            self.logger.error(f"Error in start_stop_browser: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.start_stop_button.setText("Start Browser")
            self.send_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self.screenshot_label.setText(f"Error starting browser: {str(e)}")
            self.show_error_message("Browser Error", f"Could not start Chrome browser: {str(e)}\n\nPlease make sure Google Chrome is installed on your system.")
    
    def navigate_to_url(self):
        """Navigate to the URL entered in the URL input field"""
        if not self.ensure_browser_initialized():
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
        if not self.ensure_browser_initialized():
            return
        
        try:
            # Get the visual tool from the agent
            visual_tool = self.current_agent.visual_tool
            
            # Log what we're doing
            self.logger.info(f"Executing command: {command} with parameters: {parameters}")
            
            # Execute the command
            result = visual_tool.forward(command, parameters)
            
            # Add to conversation
            self.conversation.add_message(f"Command: {command} {parameters}", is_user=True)
            self.conversation.add_message(result, is_user=False)
        except Exception as e:
            self.logger.error(f"Error executing quick command: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.conversation.add_message(f"Error executing command: {str(e)}", is_user=False)
    
    def analyze_current_page(self):
        """Analyze the current page with the selected task type"""
        if not self.ensure_browser_initialized():
            return
            
        # Get selected task
        task = self.task_selector.currentText().lower().replace(" ", "_")
        
        # Create a prompt for the selected task
        custom_prompt = f"Analyze the current webpage and {task.replace('_', ' ')}"
        
        # Add to conversation
        self.conversation.add_message(f"Analyzing page: {custom_prompt}", is_user=True)
        
        # Clear analysis area
        self.analysis_text.clear()
        self.clear_suggestions()
        
        # Find the main window to run agent in thread
        main_window = self.window()
        if hasattr(main_window, 'run_agent_in_thread'):
            self.logger.info(f"Running agent {self.current_agent_id} with analysis: {custom_prompt}")
            
            # Execute the agent in a thread
            main_window.run_agent_in_thread(
                self.current_agent_id, 
                custom_prompt,
                self.handle_analysis_result
            )
        else:
            self.logger.error("Parent window not found or missing run_agent_in_thread method")
            self.conversation.add_message("Error: Could not run analysis. Please try again.", is_user=False)
    
    def handle_analysis_result(self, result: str):
        """Handle analysis result from the agent
        
        Args:
            result: Analysis result
        """
        # Add to conversation
        self.conversation.add_message(result, is_user=False)
        
        # Set result in analysis text area
        self.analysis_text.setText(result)
        
        # Try to parse suggestions from the result
        self.parse_suggestions_from_result(result)
    
    def parse_suggestions_from_result(self, result: str):
        """Parse suggestions from the analysis result
        
        Args:
            result: Analysis result from the agent
        """
        try:
            # Look for JSON array in the response
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                suggestions = json.loads(json_str)
                
                # Clear previous suggestions
                self.clear_suggestions()
                
                if isinstance(suggestions, list):
                    # Create suggestion items
                    from app.agents.visual_web_agent import SuggestionItem
                    
                    for suggestion in suggestions:
                        item = SuggestionItem(suggestion)
                        item.clicked.connect(self.handle_suggestion_clicked)
                        self.suggestions_layout.addWidget(item)
        except:
            # No parseable JSON or other error
            pass
    
    def clear_suggestions(self):
        """Clear all suggestions from the layout"""
        # Remove all widgets from suggestions layout
        while self.suggestions_layout.count():
            item = self.suggestions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def handle_suggestion_clicked(self, suggestion_data):
        """Handle when a suggestion is clicked
        
        Args:
            suggestion_data: Suggestion data dictionary
        """
        action = suggestion_data.get("action", "").lower()
        target = suggestion_data.get("target", "")
        coordinates = suggestion_data.get("coordinates", {})
        
        try:
            # Show overlay if the agent has one
            if hasattr(self.current_agent, 'overlay'):
                screen = QApplication.primaryScreen().size()
                self.current_agent.overlay.setGeometry(0, 0, screen.width(), screen.height())
                self.current_agent.overlay.clear_highlights()
                
                if coordinates and isinstance(coordinates, dict) and "x" in coordinates and "y" in coordinates:
                    x, y = coordinates.get("x"), coordinates.get("y")
                    highlight_size = 50
                    self.current_agent.overlay.add_highlight(
                        x - highlight_size//2, 
                        y - highlight_size//2, 
                        highlight_size, 
                        highlight_size,
                        (255, 0, 0, 80),  # Red with some transparency
                        suggestion_data.get("action", "")
                    )
                    self.current_agent.overlay.show()
            
            # Log the selected action
            self.logger.info(f"Selected action: {action} on {target}")
            
            # If coordinates are available, ask for confirmation
            if coordinates and isinstance(coordinates, dict) and "x" in coordinates and "y" in coordinates:
                x, y = coordinates.get("x"), coordinates.get("y")
                
                # Ask for confirmation
                from PyQt6.QtWidgets import QMessageBox
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Confirm Action")
                msg_box.setText(f"Do you want to {action} on {target}?")
                msg_box.setInformativeText(f"This will perform an action at position ({x}, {y}).")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.No)
                
                if msg_box.exec() == QMessageBox.StandardButton.Yes:
                    # Execute the action
                    command = "click"
                    if "type" in action:
                        command = "type"
                        # Get text to type
                        from PyQt6.QtWidgets import QInputDialog
                        text, ok = QInputDialog.getText(
                            self, 
                            "Enter Text", 
                            f"What text should be typed into {target}?"
                        )
                        if ok and text:
                            # Execute command through the agent
                            if self.current_agent and hasattr(self.current_agent, 'visual_tool'):
                                self.current_agent.visual_tool.forward(command, f"{target},{text}")
                    elif "scroll" in action:
                        command = "scroll" if "down" in action else "scroll_up"
                        if self.current_agent and hasattr(self.current_agent, 'visual_tool'):
                            self.current_agent.visual_tool.forward(command, "500")
                    else:
                        # Default to click
                        if self.current_agent and hasattr(self.current_agent, 'visual_tool'):
                            self.current_agent.visual_tool.forward(command, target)
        except Exception as e:
            self.logger.error(f"Error handling suggestion: {str(e)}")
    
    def send_command(self):
        """Send a command to the agent"""
        if not self.ensure_browser_initialized():
            return
        
        # Get command text
        command = self.command_input.toPlainText().strip()
        if not command:
            return
        
        # Add to conversation
        self.conversation.add_message(command, is_user=True)
        
        # Clear input
        self.command_input.clear()
        
        # Disable UI while processing
        self.command_input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.start_stop_button.setEnabled(False)
        
        # Find the main window to run agent in thread
        main_window = self.window()
        if hasattr(main_window, 'run_agent_in_thread'):
            self.logger.info(f"Running agent {self.current_agent_id} with command: {command}")
            
            # Execute the agent in a thread
            main_window.run_agent_in_thread(
                self.current_agent_id, 
                command,
                self.handle_command_result
            )
        else:
            self.logger.error("Parent window not found or missing run_agent_in_thread method")
            self.conversation.add_message("Error: Could not run agent. Please try again.", is_user=False)
            
            # Re-enable UI
            self.command_input.setEnabled(True)
            self.send_button.setEnabled(True)
            self.start_stop_button.setEnabled(True)
    
    def handle_command_result(self, result: str):
        """Handle command result from the agent
        
        Args:
            result: Command result
        """
        # Add to conversation
        self.conversation.add_message(result, is_user=False)
        
        # Re-enable UI
        self.command_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.start_stop_button.setEnabled(True)
    
    def ensure_browser_initialized(self) -> bool:
        """Ensure the browser is properly initialized
        
        Returns:
            True if initialized, False otherwise
        """
        if self.current_agent is None or not hasattr(self.current_agent, 'browser') or self.current_agent.browser is None:
            self.show_error_message("Browser Not Started", "Please start the browser first.")
            return False
        return True
    
    def show_error_message(self, title: str, message: str):
        """Show an error message dialog
        
        Args:
            title: Dialog title
            message: Error message
        """
        QMessageBox.warning(self, title, message)
    
    def show_examples(self):
        """Show example commands for the agent"""
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
    
    def clear_input(self):
        """Clear the command input field"""
        self.command_input.clear()
    
    @pyqtSlot(object)
    def update_screenshot(self, screenshot):
        """Update the screenshot display
        
        Args:
            screenshot: Screenshot image (PIL Image)
        """
        try:
            # Convert PIL Image to QImage and then to QPixmap
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
                
                # Update browser URL in UI if available
                if self.current_agent and hasattr(self.current_agent, 'webdriver'):
                    try:
                        current_url = self.current_agent.webdriver.current_url
                        title = self.current_agent.webdriver.title
                        status_text = f"Browser at: {title} ({current_url})"
                        main_window = self.window()
                        if hasattr(main_window, 'status_bar'):
                            main_window.status_bar.showMessage(status_text, 5000)
                    except:
                        pass
        except Exception as e:
            self.logger.error(f"Error updating screenshot: {str(e)}")
