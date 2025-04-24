"""
Enhanced Visual Web Agent for sagax1
Agent for visually interacting with websites with Vision Language Model integration
"""

import os
import time
import logging
import threading
import traceback
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional, Callable, Union

from PIL import Image, ImageGrab
import pyautogui

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QScrollArea, 
    QMessageBox, QGroupBox, QGridLayout, QCheckBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QThread, QSize

from app.agents.base_agent import BaseAgent
from huggingface_hub import InferenceClient

class VisionThread(QThread):
    """Thread for handling Vision API requests."""
    
    analysis_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, api_key, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        """Initialize the Vision Thread.
        
        Args:
            api_key: Hugging Face API key
            model_id: Model ID to use
        """
        super().__init__()
        self.api_key = api_key
        self.model_id = model_id
        self.screenshot = None
        self.prompt = None
        self.running = False
    
    def set_task(self, screenshot, prompt):
        """Set the screenshot and prompt for analysis.
        
        Args:
            screenshot: PIL Image of the screen
            prompt: Prompt text for the vision model
        """
        self.screenshot = screenshot
        self.prompt = prompt
    
    def run(self):
        """Run the thread using Hugging Face's direct model endpoint."""
        self.running = True
        try:
            if not self.screenshot or not self.prompt:
                self.error_occurred.emit("No screenshot or prompt provided")
                return

            # 1) Encode the PIL Image to base64
            buf = BytesIO()
            self.screenshot.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # 2) Initialize the client
            client = InferenceClient(
                provider="nebius", 
                api_key=self.api_key
            )

            # 3) Create completion request
            completion = client.chat.completions.create(
                model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=512,
            )

            # 4) Emit the result
            analysis = completion.choices[0].message.content
            self.analysis_ready.emit(analysis)

        except Exception as e:
            err = f"Error in vision analysis: {type(e).__name__}: {e}"
            self.error_occurred.emit(err)

        finally:
            self.running = False


class OverlayWidget(QWidget):
    """Transparent overlay widget for highlighting elements on screen."""
    
    def __init__(self, parent=None):
        """Initialize the overlay widget."""
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        
        # Overlay elements
        self.highlights = []  # List of (x, y, width, height, color, text) tuples
    
    def add_highlight(self, x, y, width, height, color=(255, 0, 0, 128), text=None):
        """Add a highlight rectangle."""
        self.highlights.append((x, y, width, height, color, text))
        self.update()
    
    def clear_highlights(self):
        """Clear all highlights."""
        self.highlights = []
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event to draw highlights."""
        from PyQt6.QtGui import QPainter, QBrush, QPen, QColor
        from PyQt6.QtCore import QRect
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        for x, y, width, height, color, text in self.highlights:
            # Draw rectangle
            r, g, b, a = color
            painter.setPen(QPen(QColor(r, g, b, 255), 2))
            painter.setBrush(QBrush(QColor(r, g, b, a)))
            painter.drawRect(x, y, width, height)
            
            # Draw text if provided
            if text:
                painter.setPen(QPen(QColor(255, 255, 255, 255)))
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.drawText(QRect(x, y - 20, width, 20), Qt.AlignmentFlag.AlignCenter, text)


class SuggestionItem(QWidget):
    """Widget for displaying a suggested action."""
    
    clicked = pyqtSignal(dict)
    
    def __init__(self, suggestion_data, parent=None):
        """Initialize the suggestion item.
        
        Args:
            suggestion_data: Dictionary with suggestion data
            parent: Parent widget
        """
        super().__init__(parent)
        self.suggestion_data = suggestion_data
        
        # Set cursor to pointer to indicate clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add action type label
        self.title_label = QLabel(suggestion_data.get("action", "Action"))
        self.title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(self.title_label)
        
        # Add description label
        description = suggestion_data.get("description", "")
        self.desc_label = QLabel(description)
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)
        
        self.setLayout(layout)
        
        # Set styling based on action type
        self.set_style_by_action()
    
    def set_style_by_action(self):
        """Set the style of the item based on action type."""
        action = self.suggestion_data.get("action", "").lower()
        
        base_style = """
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
        """
        
        if "click" in action:
            self.setStyleSheet(base_style + "background-color: #e6f7ff; border: 1px solid #1890ff;")
        elif "type" in action or "enter" in action:
            self.setStyleSheet(base_style + "background-color: #f6ffed; border: 1px solid #52c41a;")
        elif "scroll" in action:
            self.setStyleSheet(base_style + "background-color: #fff7e6; border: 1px solid #faad14;")
        elif "navigate" in action or "go" in action:
            self.setStyleSheet(base_style + "background-color: #f9f0ff; border: 1px solid #722ed1;")
        else:
            self.setStyleSheet(base_style + "background-color: #f0f2f5; border: 1px solid #d9d9d9;")
    
    def mousePressEvent(self, event):
        """Handle mouse press event."""
        super().mousePressEvent(event)
        self.clicked.emit(self.suggestion_data)


class VisualWebAgent(BaseAgent):
    """Enhanced agent for visually interacting with websites using VLM"""
    
    screenshot_updated = pyqtSignal(object)
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the enhanced visual web agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
        """
        super().__init__(agent_id, config)
        
        # Configuration
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        self.device = config.get("device", "auto")
        self.api_key = config.get("api_key") or os.environ.get("HF_API_KEY")
        
        if not self.api_key:
            self.logger.warning("No API key provided. Vision features will not work.")
        
        # Initialize vision thread
        self.vision_thread = VisionThread(self.api_key, self.model_id)
        self.vision_thread.analysis_ready.connect(self.handle_analysis_result)
        self.vision_thread.error_occurred.connect(self.handle_analysis_error)
        
        # State variables
        self.current_screenshot = None
        self.analysis_text = ""
        self.suggestions = []
        
        # Store generated media paths
        self.generated_media = []
        
        # Create overlay widget
        self.overlay = OverlayWidget()
        
        # Initialize browser components from legacy agent
        self.browser = None
        self.webdriver = None
        self.is_initialized = False
        
        # Create browser thread for automated screenshots
        from app.agents.visual_web_agent import BrowserThread
        self.browser_thread = BrowserThread(self)
        self.browser_thread.screenshot_ready.connect(self._on_screenshot_ready)
    
    def initialize(self) -> None:
        """Initialize the agent with the necessary components"""
        if self.is_initialized:
            return
            
        try:
            # Initialize the browser as in the original visual web agent
            from smolagents import Tool, CodeAgent
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            import helium
            
            self.logger.info("Initializing enhanced visual web agent")
            
            # Configure browser options
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--force-device-scale-factor=1")
            chrome_options.add_argument("--window-size=1000,800")
            chrome_options.add_argument("--disable-pdf-viewer")
            chrome_options.add_argument("--window-position=0,0")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Force browser to be headless for more reliability
            chrome_options.add_argument("--headless=new")
            
            # Start the browser
            service = Service(ChromeDriverManager().install())
            self.webdriver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Initialize helium
            helium.set_driver(self.webdriver)
            self.browser = helium
            
            # Navigate to a simple page
            self.webdriver.get("https://www.google.com")
            
            # Start browser thread for regular screenshots
            self.browser_thread.start()
            
            self.is_initialized = True
            self.logger.info(f"Enhanced visual web agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced visual web agent: {str(e)}")
            traceback.print_exc()
            raise
    
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
            # Update progress if callback is provided
            if callback:
                callback("Processing your visual web request...")
            
            # Check what kind of task is being requested
            task_type = self._determine_task_type(input_text)
            
            if task_type == "browser_action":
                # Use original browser automation capabilities
                return self._handle_browser_action(input_text, callback)
            else:
                # Use VLM for analysis
                return self._handle_vlm_analysis(input_text, task_type, callback)
            
        except Exception as e:
            error_msg = f"Error running enhanced visual web agent: {str(e)}"
            self.logger.error(error_msg)
            traceback.print_exc()
            return f"Sorry, I encountered an error while processing your request: {error_msg}"
    
    def _determine_task_type(self, input_text: str) -> str:
        """Determine the type of task requested
        
        Args:
            input_text: Input text from the user
            
        Returns:
            Task type identifier
        """
        input_lower = input_text.lower()
        
        # Check for browser action commands
        if any(term in input_lower for term in ["go to", "navigate to", "click", "type", "scroll", "back"]):
            return "browser_action"
        
        # Check for specific analysis tasks
        if "explain" in input_lower or "what's on" in input_lower or "describe" in input_lower:
            return "explain_content"
        
        if "find" in input_lower or "locate" in input_lower or "where is" in input_lower:
            return "find_elements"
        
        if "suggest" in input_lower or "what should" in input_lower:
            return "suggest_actions"
        
        if "guide" in input_lower or "help me navigate" in input_lower:
            return "guide_navigation"
        
        # Default to general analysis
        return "general_analysis"
    
    def _handle_browser_action(self, input_text: str, callback: Optional[Callable[[str], None]]) -> str:
        """Handle traditional browser automation actions
        
        Args:
            input_text: Input text from the user
            callback: Optional callback for progress updates
            
        Returns:
            Result of the browser action
        """
        # Use existing browser automation from original agent
        # This preserves the existing functionality
        
        from app.agents.visual_web_agent import VisualWebAutomationTool
        
        # Only create the tool if needed
        visual_tool = VisualWebAutomationTool()
        visual_tool.browser = self.browser
        visual_tool.webdriver = self.webdriver
        
        # Parse the command from input
        command, parameters = self._parse_browser_command(input_text)
        
        # Progress update
        if callback:
            callback(f"Executing browser command: {command} {parameters}")
        
        # Execute command
        result = visual_tool.forward(command, parameters)
        
        # Take a screenshot to show the result
        self._capture_screenshot()
        
        # Format the response to include both the result and a mention of the screenshot
        response = f"{result}\n\nI've captured a screenshot of the current state. You can see it in the display area."
        
        # Add to history
        self.add_to_history(input_text, response)
        
        return response
    
    def _parse_browser_command(self, input_text: str) -> tuple:
        """Parse a browser command from input text
        
        Args:
            input_text: Input text from the user
            
        Returns:
            Tuple of (command, parameters)
        """
        input_lower = input_text.lower()
        
        # Handle navigation commands
        if "go to " in input_lower or "navigate to " in input_lower:
            url = input_lower.split("go to ")[-1].split("navigate to ")[-1].strip()
            if not url.startswith("http"):
                url = "https://" + url
            return "go_to", url
        
        # Handle click commands
        if "click " in input_lower:
            element = input_lower.split("click ")[-1].strip()
            return "click", element
        
        # Handle typing commands
        if "type " in input_lower:
            parts = input_text.split("type ")[-1].strip()
            if " into " in parts:
                text, element = parts.split(" into ", 1)
                return "type", f"{element},{text}"
            else:
                return "type", parts
        
        # Handle scrolling
        if "scroll down" in input_lower:
            amount = "500"  # Default amount
            if "by " in input_lower:
                try:
                    amount = input_lower.split("by ")[-1].split()[0]
                except:
                    pass
            return "scroll", amount
            
        if "scroll up" in input_lower:
            amount = "500"  # Default amount
            if "by " in input_lower:
                try:
                    amount = input_lower.split("by ")[-1].split()[0]
                except:
                    pass
            return "scroll_up", amount
        
        # Handle back navigation
        if "go back" in input_lower or "back" in input_lower:
            return "back", ""
        
        # Default to screenshot
        return "screenshot", ""
    
    def _handle_vlm_analysis(self, input_text: str, task_type: str, callback: Optional[Callable[[str], None]]) -> str:
        """Handle VLM-based analysis tasks
        
        Args:
            input_text: Input text from the user
            task_type: Type of analysis to perform
            callback: Optional callback for progress updates
            
        Returns:
            Analysis result
        """
        # Make sure we have a screenshot
        if not self.current_screenshot:
            self._capture_screenshot()
        
        if not self.current_screenshot:
            return "Unable to capture a screenshot. Please make sure the browser is running."
        
        # Create a prompt based on the task type
        prompt = self._create_prompt_for_task(input_text, task_type)
        
        # Progress update
        if callback:
            callback("Analyzing the webpage with AI vision model...")
        
        # Set up the thread for vision analysis
        self.vision_thread.set_task(self.current_screenshot, prompt)
        
        # Since this is synchronous, we need to use a special approach to make it work
        # with the existing agent structure
        from threading import Event
        
        result_ready = Event()
        analysis_result = ["Analysis not available"]
        
        # Create a temporary slot to handle the result
        def temp_handle_result(result):
            analysis_result[0] = result
            result_ready.set()
        
        # Connect the signal temporarily
        self.vision_thread.analysis_ready.connect(temp_handle_result)
        
        # Start the analysis
        self.vision_thread.start()
        
        # Wait for the result (with timeout)
        result_ready.wait(timeout=30)
        
        # Disconnect the signal
        self.vision_thread.analysis_ready.disconnect(temp_handle_result)
        
        # Format the result
        result = analysis_result[0]
        
        # Add to history
        self.add_to_history(input_text, result)
        
        return result
    
    def _create_prompt_for_task(self, input_text: str, task_type: str) -> str:
        """Create a prompt for the vision model based on the task type
        
        Args:
            input_text: Input text from the user
            task_type: Type of analysis to perform
            
        Returns:
            Prompt for the vision model
        """
        base_prompt = "You are an AI assistant helping a user navigate a web browser. "
        
        if task_type == "suggest_actions":
            prompt = base_prompt + """
            Look at this screenshot of a webpage and suggest the most useful actions the user could take.
            
            For each action, provide:
            1. A clear description of what the action does
            2. The exact element to interact with
            3. What the expected outcome would be
            
            Format your suggestions as a JSON array with objects that have these properties:
            - action: The type of action (click, type, scroll, navigate, etc.)
            - target: The element to interact with (be specific)
            - description: A helpful description of what this action will do
            - coordinates (optional): If you can identify the element location, provide x,y coordinates
            
            Example format:
            ```json
            [
                {
                    "action": "click",
                    "target": "Sign in button",
                    "description": "Click the blue Sign In button in the top right to log in",
                    "coordinates": {"x": 920, "y": 80}
                }
            ]
            ```
            
            Provide 3-5 of the most relevant suggestions based on the current page context.
            Also include 2-3 sentences of general information about what page the user is on.
            """
        
        elif task_type == "explain_content":
            prompt = base_prompt + """
            Look at this screenshot and explain:
            
            1. What website/page is this?
            2. What is the main content or purpose of this page?
            3. Key sections or features visible on the page
            4. Any important information the user should notice
            
            Provide a clear, concise explanation that would help someone understand what they're looking at.
            """
        
        elif task_type == "find_elements":
            prompt = base_prompt + """
            I need to find important interactive elements on this webpage. Please:
            
            1. Identify the main interactive elements (buttons, links, forms, etc.)
            2. For each element, provide:
               - A description of what it is
               - Its approximate position on screen (x,y coordinates if possible)
               - What action it would perform if clicked/used
            
            Format your findings in a clear, structured way that makes it easy to understand where everything is on the page.
            """
        
        elif task_type == "guide_navigation":
            prompt = base_prompt + """
            I'm looking at this webpage and need guidance on how to navigate it effectively.
            
            Please provide:
            
            1. An overview of what website/page I'm on
            2. The main navigation options available
            3. A step-by-step guide for the most common task a user would want to do on this page
            4. Any tips for navigating this site effectively
            
            Make your guidance clear, concise, and easy to follow for someone who may be unfamiliar with this website.
            """
        
        else:
            # General analysis - use the user's input directly
            prompt = base_prompt + f"""
            Please analyze this screenshot of a webpage based on this request: "{input_text}"
            
            Provide a detailed and helpful response that addresses the specific question or request.
            Include relevant details about what's visible on the page that relate to the request.
            """
        
        return prompt
    
    def _capture_screenshot(self) -> bool:
        """Capture a screenshot of the current browser window
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.webdriver:
                self.logger.error("Browser not initialized")
                return False
            
            # Capture screenshot using webdriver
            screenshot_bytes = self.webdriver.get_screenshot_as_png()
            
            # Convert to PIL Image
            from PIL import Image
            from io import BytesIO
            self.current_screenshot = Image.open(BytesIO(screenshot_bytes))
            
            # Emit signal for UI update
            self.screenshot_updated.emit(self.current_screenshot)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error capturing screenshot: {str(e)}")
            return False
    
    def handle_analysis_result(self, analysis):
        """Handle the vision analysis result.
        
        Args:
            analysis: Analysis text from the vision model
        """
        self.analysis_text = analysis
        
        # Parse for suggestions if appropriate
        try:
            import json
            
            # Look for JSON array in the response
            json_start = analysis.find('[')
            json_end = analysis.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = analysis[json_start:json_end]
                suggestions = json.loads(json_str)
                
                if isinstance(suggestions, list):
                    self.suggestions = suggestions
        except:
            # Not a JSON response or error parsing
            self.suggestions = []
    
    def handle_analysis_error(self, error_message):
        """Handle vision analysis error.
        
        Args:
            error_message: Error message
        """
        self.logger.error(f"Vision analysis error: {error_message}")
        self.analysis_text = f"Error: {error_message}"
    
    def _on_screenshot_ready(self, screenshot):
        """Handle screenshot ready signal
        
        Args:
            screenshot: New screenshot
        """
        self.current_screenshot = screenshot
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
        self.current_screenshot = None
        self.analysis_text = ""
        self.suggestions = []
    
    def cleanup(self) -> None:
        """Clean up resources"""
        # Stop browser thread
        if hasattr(self, 'browser_thread') and self.browser_thread.isRunning():
            self.browser_thread.running = False
            self.browser_thread.wait()
        
        # Close browser
        if hasattr(self, 'browser') and self.browser is not None:
            try:
                import helium
                helium.kill_browser()
                self.browser = None
                self.webdriver = None
            except Exception as e:
                self.logger.error(f"Error closing browser: {str(e)}")
        
        # Close overlay
        if hasattr(self, 'overlay') and self.overlay is not None:
            self.overlay.close()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        return [
            "web_automation",
            "visual_analysis",
            "screenshot_capture",
            "element_recognition",
            "navigation_guidance",
            "action_suggestions",
            "vlm_integration"
        ]
