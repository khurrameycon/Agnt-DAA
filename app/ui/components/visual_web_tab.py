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
        # Make sure we have a browser
        if self.current_agent is None or not hasattr(self.current_agent, 'visual_tool') or self.current_agent.visual_tool.browser is None:
            self.logger.info("Browser not active, attempting to start...")
            # Use start_stop_browser instead of toggle_browser
            self.start_stop_browser()
            
            # Wait for browser to start
            import time
            time.sleep(3)
            
            # Check again
            if self.current_agent is None or not hasattr(self.current_agent, 'visual_tool') or self.current_agent.visual_tool.browser is None:
                self.logger.error("Browser failed to start")
                self.show_error_message("Browser Not Started", "Please start the browser first.")
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
    
    def ensure_browser_initialized(self):
        """Ensure the browser is properly initialized"""
        if self.current_agent is None or not hasattr(self.current_agent, 'visual_tool') or self.current_agent.visual_tool.browser is None:
            # Try to start the browser
            if self.start_stop_button.text() == "Start Browser":
                self.start_stop_browser()
                # Wait a moment for the browser to start
                import time
                time.sleep(2)
                
            # Check again
            if self.current_agent is None or not hasattr(self.current_agent, 'visual_tool') or self.current_agent.visual_tool.browser is None:
                self.show_error_message("Browser Not Started", "Please start the browser first.")
                return False
        return True
    
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
            return
        
        try:
            # Store the selected agent ID immediately to avoid empty ID issues
            self.current_agent_id = agent_id
            
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
                self.current_agent = agent
                
                # Enable buttons
                self.start_stop_button.setEnabled(True)
                
                # Enable send button only if browser is started
                self.send_button.setEnabled(hasattr(agent, 'visual_tool') and 
                                         agent.visual_tool.browser is not None)
                
                # Update button text
                if hasattr(agent, 'visual_tool') and agent.visual_tool.browser is not None:
                    self.start_stop_button.setText("Stop Browser")
                else:
                    self.start_stop_button.setText("Start Browser")
            else:
                # Store ID but not the agent instance yet
                self.current_agent = None
                
                # Enable start button
                self.start_stop_button.setEnabled(True)
                self.start_stop_button.setText("Start Browser")
                
                # Disable send button until browser is started
                self.send_button.setEnabled(False)
                
        except Exception as e:
            self.logger.error(f"Error selecting agent: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
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
    

    def init_browser_manually(self):
        """Manually initialize the browser as a fallback method"""
        if self.current_agent is None:
            self.logger.error("No agent available for manual browser initialization")
            return False
            
        try:
            self.logger.info("Attempting manual browser initialization...")
            
            # Import required libraries
            import helium
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            
            # Configure browser options
            options = webdriver.ChromeOptions()
            options.add_argument("--force-device-scale-factor=1")
            options.add_argument("--window-size=1000,800")
            options.add_argument("--disable-pdf-viewer")
            options.add_argument("--window-position=0,0")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            # Start Chrome directly
            self.logger.info("Starting Chrome directly...")
            browser_driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), 
                options=options
            )
            
            # Navigate to a page to verify it works
            browser_driver.get("https://www.google.com")
            self.logger.info(f"Browser navigated to: {browser_driver.current_url}")
            
            # Set the driver in helium
            helium.set_driver(browser_driver)
            
            # Set the browser in visual_tool
            if hasattr(self.current_agent, 'visual_tool'):
                self.current_agent.visual_tool.browser = helium
                self.current_agent.visual_tool.webdriver = browser_driver
                
                # Take a screenshot to verify
                screenshot_bytes = browser_driver.get_screenshot_as_png()
                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(screenshot_bytes))
                self.update_screenshot(image)
                
                self.logger.info("Manual browser initialization successful")
                self.start_stop_button.setText("Stop Browser")
                self.send_button.setEnabled(True)
                return True
            else:
                self.logger.error("Agent doesn't have visual_tool attribute")
                return False
                
        except Exception as e:
            self.logger.error(f"Manual browser initialization failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


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
            
            # Try normal initialization first
            try:
                self.logger.info("Initializing agent with standard method...")
                agent.initialize()
                
                # Wait for browser to potentially start
                import time
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Standard initialization failed: {str(e)}")
            
            # Check if browser started
            if hasattr(agent, 'visual_tool') and agent.visual_tool.browser is not None:
                self.logger.info("Browser started with standard method")
                self.start_stop_button.setText("Stop Browser")
                self.send_button.setEnabled(True)
                self.conversation.add_message("Browser started successfully", is_user=False)
                return
            
            # If standard method failed, try manual initialization
            self.logger.info("Standard browser initialization failed, trying manual method...")
            if self.init_browser_manually():
                self.logger.info("Manual browser initialization successful")
                return
            
            # If both methods fail
            self.start_stop_button.setText("Start Browser")
            self.send_button.setEnabled(False)
            self.screenshot_label.setText("Failed to start browser")
            self.show_error_message("Browser Error", "Failed to start browser. Please check if Chrome is installed and try again.")
                
        except Exception as e:
            self.logger.error(f"Error in start_stop_browser: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.start_stop_button.setText("Start Browser")
            self.send_button.setEnabled(False)
            self.screenshot_label.setText(f"Error starting browser: {str(e)}")
            self.show_error_message("Browser Error", f"Could not start Chrome browser: {str(e)}\n\nPlease make sure Google Chrome is installed on your system.")
    
    def send_command(self):
        """Send command to agent"""
        try:
            # Log that the function was called
            self.logger.info("Send command method called")
            
            # Check if browser is running
            if self.current_agent is None or not hasattr(self.current_agent, 'visual_tool') or self.current_agent.visual_tool.browser is None:
                self.logger.error("Browser not started")
                self.show_error_message("Browser Not Started", "Please start the browser first.")
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
            
            # Disable UI while processing
            self.command_input.setEnabled(False)
            self.send_button.setEnabled(False)
            self.start_stop_button.setEnabled(False)
            
            # Show processing message
            self.conversation.add_message("Processing your request...", is_user=False)
            
            # Try simple direct execution first for certain patterns
            if any(pattern in command.lower() for pattern in ["go to ", "navigate to "]):
                try:
                    # Extract URL
                    url = None
                    if "go to " in command.lower():
                        url = command.lower().split("go to ")[1].split()[0].strip()
                    elif "navigate to " in command.lower():
                        url = command.lower().split("navigate to ")[1].split()[0].strip()
                    
                    if url:
                        # Make sure URL has protocol
                        if not url.startswith("http"):
                            url = "https://" + url
                            
                        # Execute directly
                        self.logger.info(f"Direct execution: Navigating to {url}")
                        result = self.current_agent.visual_tool.forward("go_to", url)
                        self.conversation.add_message(f"Navigated to {url}", is_user=False)
                        
                        # Re-enable UI
                        self.command_input.setEnabled(True)
                        self.send_button.setEnabled(True)
                        self.start_stop_button.setEnabled(True)
                        return
                except Exception as e:
                    self.logger.error(f"Error in direct execution: {str(e)}")
                    # Continue to regular agent processing
            
            # Find the main window to run agent in thread
            main_window = self.window()
            if hasattr(main_window, 'run_agent_in_thread'):
                self.logger.info(f"Running agent {self.current_agent_id} with command: {command}")
                
                # Execute the agent in a thread
                main_window.run_agent_in_thread(
                    self.current_agent_id, 
                    command,
                    self.handle_agent_result
                )
            else:
                self.logger.error("Parent window not found or missing run_agent_in_thread method")
                self.conversation.add_message("Error: Could not run agent. Please try again.", is_user=False)
                
                # Re-enable UI
                self.command_input.setEnabled(True)
                self.send_button.setEnabled(True)
                self.start_stop_button.setEnabled(True)
                
        except Exception as e:
            self.logger.error(f"Error in send_command: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Show error
            self.conversation.add_message(f"Error: {str(e)}", is_user=False)
            
            # Re-enable UI
            self.command_input.setEnabled(True)
            self.send_button.setEnabled(True)
            self.start_stop_button.setEnabled(True)
    
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
            # Suggest a more compatible model
            from PyQt6.QtWidgets import QMessageBox
            result = QMessageBox.question(
                self,
                "Model Recommendation",
                "For best compatibility with visual web automation, we recommend using 'meta-llama/Llama-3.2-3B-Instruct'.\n\n"
                "Would you like to use this model?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            use_recommended = result == QMessageBox.StandardButton.Yes
            
            # Call the create method with the recommended model flag
            main_window.create_visual_web_agent(use_recommended_model=use_recommended)
        else:
            # Fallback if we can't find the method
            QMessageBox.warning(
                self,
                "Not Implemented",
                "Create visual web agent functionality not found in main window."
            )