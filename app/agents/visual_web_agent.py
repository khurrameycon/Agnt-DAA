"""
Visual Web Automation Agent for SagaX1
Agent for visually interacting with websites through screenshots and automation
"""

import os
import time
import logging
import threading
import tempfile
from io import BytesIO
from typing import Dict, Any, List, Optional, Callable

from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread

from app.agents.base_agent import BaseAgent
from smolagents import CodeAgent, Tool, DuckDuckGoSearchTool, tool

class VisualWebAutomationTool(Tool):
    """Tool for visual web automation"""
    
    name = "visual_web_automation"
    description = "Execute web automation commands using a browser"
    inputs = {
        "command": {
            "type": "string", 
            "description": "The command to execute (go_to, click, type, screenshot, scroll, back, etc.)"
        },
        "parameters": {
            "type": "string", 
            "description": "Parameters for the command, such as URL, element selector, text, etc."
        }
    }
    output_type = "string"
    
    def __init__(self):
        """Initialize the visual web automation tool"""
        super().__init__()
        self.browser = None
        self.webdriver = None
        self.logger = logging.getLogger(__name__)
    
    def _ensure_browser_started(self):
        """Ensure the browser is started"""
        if self.browser is None:
            try:
                import helium
                from selenium import webdriver
                
                # Configure browser options
                options = webdriver.ChromeOptions()
                options.add_argument("--force-device-scale-factor=1")
                options.add_argument("--window-size=1280,1024")
                options.add_argument("--disable-pdf-viewer")
                
                # Start the browser
                self.browser = helium.start_chrome(headless=False, options=options)
                self.webdriver = helium.get_driver()
                self.logger.info("Browser started successfully")
                
            except Exception as e:
                self.logger.error(f"Error starting browser: {str(e)}")
                raise
    
    def forward(self, command: str, parameters: str) -> str:
        """Execute a web automation command
        
        Args:
            command: Command to execute
            parameters: Parameters for the command
            
        Returns:
            Result of the command execution
        """
        try:
            self._ensure_browser_started()
            
            # Import helium here to ensure it's only imported when needed
            import helium
            from selenium.webdriver.common.by import By
            from selenium.webdriver.common.keys import Keys
            
            command = command.lower().strip()
            
            # Execute the command based on the command type
            if command == "go_to":
                url = parameters
                if not url.startswith("http"):
                    url = "https://" + url
                helium.go_to(url)
                return f"Navigated to {url}"
                
            elif command == "click":
                element = parameters
                helium.click(element)
                return f"Clicked on {element}"
                
            elif command == "type":
                # Split parameters into target and text
                parts = parameters.split(',', 1)
                if len(parts) != 2:
                    return "Error: Type command requires parameters in format 'element,text'"
                
                element = parts[0].strip()
                text = parts[1].strip()
                
                # Type the text
                helium.write(text, into=element)
                return f"Typed '{text}' into {element}"
                
            elif command == "screenshot":
                # Take a screenshot
                screenshot = self.webdriver.get_screenshot_as_png()
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file.write(screenshot)
                    return f"Screenshot saved to {temp_file.name}"
                
            elif command == "scroll":
                try:
                    pixels = int(parameters)
                    script = f"window.scrollBy(0, {pixels});"
                    self.webdriver.execute_script(script)
                    return f"Scrolled by {pixels} pixels"
                except ValueError:
                    return f"Error: Scroll parameter must be an integer, got '{parameters}'"
                
            elif command == "back":
                helium.go_back()
                return "Navigated back"
                
            elif command == "forward":
                self.webdriver.forward()
                return "Navigated forward"
                
            elif command == "get_text":
                element = parameters
                text = helium.Text(element).web_element.text
                return f"Text from {element}: {text}"
                
            elif command == "current_url":
                return f"Current URL: {self.webdriver.current_url}"
                
            elif command == "page_source":
                # Get the page source but truncate if too long
                source = self.webdriver.page_source
                if len(source) > 10000:
                    source = source[:10000] + "... [truncated]"
                return f"Page source:\n{source}"
                
            else:
                return f"Unknown command: {command}"
                
        except Exception as e:
            self.logger.error(f"Error executing web automation command: {str(e)}")
            return f"Error: {str(e)}"

class ScreenshotTool(Tool):
    """Tool for taking screenshots of the current browser window"""
    
    name = "take_screenshot"
    description = "Take a screenshot of the current browser window"
    inputs = {}
    output_type = "image"
    
    def __init__(self, visual_tool: VisualWebAutomationTool):
        """Initialize the screenshot tool
        
        Args:
            visual_tool: The visual web automation tool to use
        """
        super().__init__()
        self.visual_tool = visual_tool
        self.logger = logging.getLogger(__name__)
    
    def forward(self) -> Image.Image:
        """Take a screenshot
        
        Returns:
            Screenshot image
        """
        try:
            self.visual_tool._ensure_browser_started()
            
            # Take screenshot
            screenshot_bytes = self.visual_tool.webdriver.get_screenshot_as_png()
            
            # Convert to PIL Image
            return Image.open(BytesIO(screenshot_bytes))
            
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {str(e)}")
            # Return a blank image with error message
            img = Image.new('RGB', (800, 100), color='red')
            return img

class BrowserThread(QThread):
    """Thread for running the browser"""
    
    screenshot_ready = pyqtSignal(object)
    
    def __init__(self, agent):
        """Initialize the browser thread
        
        Args:
            agent: The agent to run
        """
        super().__init__()
        self.agent = agent
        self.running = False
        self.interval = 2  # seconds
    
    def run(self):
        """Run the thread"""
        self.running = True
        
        while self.running:
            # Take screenshot if browser is running
            if self.agent.visual_tool.browser is not None:
                try:
                    screenshot = self.agent.screenshot_tool.forward()
                    self.screenshot_ready.emit(screenshot)
                except Exception as e:
                    self.agent.logger.error(f"Error taking screenshot: {str(e)}")
            
            # Sleep for interval
            time.sleep(self.interval)
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()

class VisualWebAgent(BaseAgent):
    """Agent for visually interacting with websites through screenshots and automation"""
    
    screenshot_updated = pyqtSignal(object)
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the visual web agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                max_tokens: Maximum number of tokens to generate
                temperature: Temperature for generation
        """
        BaseAgent.__init__(self, agent_id, config)
        QObject.__init__(self)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3-8B-Instruct")
        self.device = config.get("device", "auto")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        self.authorized_imports = config.get("authorized_imports", [])
        
        # Initialize tools
        self.visual_tool = VisualWebAutomationTool()
        self.screenshot_tool = ScreenshotTool(self.visual_tool)
        
        self.agent = None
        self.is_initialized = False
        
        # Initialize browser thread
        self.browser_thread = BrowserThread(self)
        self.browser_thread.screenshot_ready.connect(self._on_screenshot_ready)
    
    def _on_screenshot_ready(self, screenshot):
        """Handle screenshot ready
        
        Args:
            screenshot: Screenshot image
        """
        self.screenshot_updated.emit(screenshot)
    
    def initialize(self) -> None:
        """Initialize the model and agent"""
        if self.is_initialized:
            return
        
        try:
            from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
            
            self.logger.info(f"Initializing visual web agent with model {self.model_id}")
            
            # Try to create the model based on the model_id
            try:
                # First try TransformersModel for local models
                model = TransformersModel(
                    model_id=self.model_id,
                    device_map=self.device,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    trust_remote_code=True
                )
                self.logger.info(f"Using TransformersModel for {self.model_id}")
            except Exception as e:
                self.logger.warning(f"Failed to load model with TransformersModel: {str(e)}")
                
                # Try HfApiModel for API-based models
                try:
                    model = HfApiModel(
                        model_id=self.model_id,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    self.logger.info(f"Using HfApiModel for {self.model_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to load model with HfApiModel: {str(e)}")
                    
                    # Fallback to OpenAI-compatible API
                    try:
                        model = OpenAIServerModel(
                            model_id=self.model_id,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature
                        )
                        self.logger.info(f"Using OpenAIServerModel for {self.model_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load model with OpenAIServerModel: {str(e)}")
                        
                        # Final fallback to LiteLLM
                        model = LiteLLMModel(
                            model_id=self.model_id,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature
                        )
                        self.logger.info(f"Using LiteLLMModel for {self.model_id}")
            
            # Initialize tools
            tools = self._initialize_tools()
            
            # Create the agent
            self.agent = CodeAgent(
                tools=tools,
                model=model,
                additional_authorized_imports=["helium", "selenium", "time"] + self.authorized_imports,
                verbosity_level=1
            )
            
            # Start browser thread
            self.browser_thread.start()
            
            self.is_initialized = True
            self.logger.info(f"Visual web agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing visual web agent: {str(e)}")
            raise
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize tools for the agent
        
        Returns:
            List of tools
        """
        tools = []
        
        # Add web search tool
        tools.append(DuckDuckGoSearchTool())
        
        # Add visual web automation tool
        tools.append(self.visual_tool)
        
        # Add screenshot tool
        tools.append(self.screenshot_tool)
        
        return tools
    
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
            # Enhance the prompt with visual web automation guidance
            enhanced_prompt = f"""
You are a visual web automation agent that can control a web browser to accomplish tasks.
You can see screenshots of the browser window and interact with it using commands.

You have access to these tools:
- web_search: Search the web for information
- visual_web_automation: Control the browser with commands like:
  * go_to: Navigate to a URL (add https:// if missing)
  * click: Click on an element (button, link, etc.)
  * type: Type text into a field (format: "element,text")
  * screenshot: Take a screenshot
  * scroll: Scroll by a number of pixels
  * back: Go back to the previous page
  * get_text: Get text from an element
  * current_url: Get the current URL
- take_screenshot: Take a screenshot and view it

After each web automation action, wait briefly to let the page load.

USER TASK: {input_text}

First, take a screenshot to see the current state of the browser.
"""
            
            # Run the agent
            result = self.agent.run(enhanced_prompt)
            
            # Add to history
            self.add_to_history(input_text, str(result))
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Error running visual web agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while automating the web browser: {error_msg}"
    
    def reset(self) -> None:
        """Reset the agent's state"""
        if self.agent:
            # Reset the agent's memory
            self.agent.memory.reset()
        
        self.clear_history()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        return ["web_search", "web_automation", "visual_interaction", "screenshot"]
    
    def cleanup(self) -> None:
        """Clean up resources"""
        # Stop browser thread
        if hasattr(self, 'browser_thread') and self.browser_thread.isRunning():
            self.browser_thread.stop()
        
        # Close browser
        if hasattr(self, 'visual_tool') and self.visual_tool.browser is not None:
            try:
                import helium
                helium.kill_browser()
                self.visual_tool.browser = None
                self.visual_tool.webdriver = None
            except Exception as e:
                self.logger.error(f"Error closing browser: {str(e)}")