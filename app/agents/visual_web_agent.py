"""
Visual Web Automation Agent for sagax1
Agent for visually interacting with websites through screenshots and automation
Enhanced with features from the web_browser.ipynb example
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

# Add these imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import traceback

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
        """Ensure the browser is started with optimal settings for visual automation"""
        if self.browser is None:
            try:
                self.logger.info("Starting browser with improved settings...")
                
                # Configure browser options
                options = webdriver.ChromeOptions()
                options.add_argument("--force-device-scale-factor=1")
                options.add_argument("--window-size=1000,800")
                options.add_argument("--disable-pdf-viewer")
                options.add_argument("--window-position=0,0")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                
                # Force browser to be visible
                options.add_argument("--headless=new")  # Use new headless mode
                
                # Use WebDriver Manager to get the correct ChromeDriver
                service = Service(ChromeDriverManager().install())
                
                # Start the browser using Selenium
                self.logger.info("Creating Chrome driver...")
                self.webdriver = webdriver.Chrome(service=service, options=options)
                
                # Initialize helium with the driver
                import helium
                self.logger.info("Setting Helium driver...")
                helium.set_driver(self.webdriver)
                self.browser = helium
                
                # Navigate to a simple page to verify browser is working
                self.webdriver.get("https://www.google.com")
                self.logger.info(f"Browser started and navigated to: {self.webdriver.current_url}")
                
            except Exception as e:
                self.logger.error(f"Error starting browser: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
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
                
            elif command == "click_link":
                # For more reliable link clicking
                element = parameters
                helium.click(helium.Link(element))
                return f"Clicked on link '{element}'"
                
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
                    # Using helium's scroll function
                    helium.scroll_down(num_pixels=pixels)
                    return f"Scrolled down by {pixels} pixels"
                except ValueError:
                    return f"Error: Scroll parameter must be an integer, got '{parameters}'"
                
            elif command == "scroll_up":
                try:
                    pixels = int(parameters)
                    # Using helium's scroll function
                    helium.scroll_up(num_pixels=pixels)
                    return f"Scrolled up by {pixels} pixels"
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
                
            elif command == "close_popups":
                # Improved popup handling based on the notebook
                webdriver.ActionChains(self.webdriver).send_keys(Keys.ESCAPE).perform()
                return "Attempted to close popups using Escape key"
                
            elif command == "search_text":
                # Implement Ctrl+F search functionality from the notebook
                parts = parameters.split(',', 1)
                text = parts[0].strip()
                nth_result = 1
                if len(parts) > 1 and parts[1].strip().isdigit():
                    nth_result = int(parts[1].strip())
                
                elements = self.webdriver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
                if not elements:
                    return f"No matches found for '{text}'"
                if nth_result > len(elements):
                    return f"Match n°{nth_result} not found (only {len(elements)} matches found)"
                
                result = f"Found {len(elements)} matches for '{text}'."
                elem = elements[nth_result - 1]
                self.webdriver.execute_script("arguments[0].scrollIntoView(true);", elem)
                result += f" Focused on element {nth_result} of {len(elements)}"
                return result
                
            elif command == "check_exists":
                # Check if an element exists
                element = parameters
                try:
                    if hasattr(helium, element):  # If it's a type like Text, Link, etc.
                        exists = getattr(helium, element)().exists()
                    else:
                        # Assume it's a text element
                        exists = helium.Text(element).exists()
                    
                    if exists:
                        return f"Element '{element}' exists on the page"
                    else:
                        return f"Element '{element}' does not exist on the page"
                except Exception as e:
                    return f"Error checking if element exists: {str(e)}"
                
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
            self.logger.error(f"Error in forward method: {str(e)}")
            self.logger.error(traceback.format_exc())
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

class VisualWebSignals(QObject):
    """Signal handler for VisualWebAgent"""
    screenshot_updated = pyqtSignal(object)


class VisualWebAgent(BaseAgent):
    """Agent for visually interacting with websites through screenshots and automation"""
    
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
        super().__init__(agent_id, config)
        
        # Create signal handler
        self.signals = VisualWebSignals()
        self.screenshot_updated = self.signals.screenshot_updated
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3.2-3B-Instruct")
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
        self.signals.screenshot_updated.emit(screenshot)
    
    def initialize(self) -> None:
        """Initialize the model and agent"""
        if self.is_initialized:
            return
        
        try:
            from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
            from smolagents import CodeAgent, Tool
            import helium
            import selenium
            self.logger.info(f"Initializing visual web agent with model {self.model_id}")
            
            # Try to create the model based on the model_id
            try:
                # First try TransformersModel for local models
                model = TransformersModel(
                    model_id=self.model_id,
                    device_map=self.device,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    trust_remote_code=True,
                    do_sample=True
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
            
            # Define screenshot callback for agent memory
            # In app/agents/visual_web_agent.py, find the save_screenshot_to_memory function:

            def save_screenshot_to_memory(memory_step, agent):
                """Save screenshot to agent memory step"""
                try:
                    # Wait a bit for page to load
                    time.sleep(1.0)
                    
                    # Get screenshot
                    if self.visual_tool.browser is not None:
                        # Remove previous screenshots for lean processing
                        current_step = memory_step.step_number
                        for previous_memory_step in agent.memory.steps:
                            if hasattr(previous_memory_step, 'step_number') and previous_memory_step.step_number <= current_step - 2:
                                if hasattr(previous_memory_step, 'observations_images'):
                                    previous_memory_step.observations_images = None
                        
                        # Take screenshot
                        try:
                            screenshot_bytes = self.visual_tool.webdriver.get_screenshot_as_png()
                            image = Image.open(BytesIO(screenshot_bytes))
                            
                            # Set screenshot in memory step
                            memory_step.observations_images = [image.copy()]
                            
                            # Update observations with current URL
                            url_info = f"Current url: {self.visual_tool.webdriver.current_url}"
                            memory_step.observations = (
                                url_info if memory_step.observations is None 
                                else memory_step.observations + "\n" + url_info
                            )
                            
                            self.logger.info("Screenshot saved to memory successfully")
                        except Exception as screenshot_error:
                            self.logger.error(f"Error taking screenshot: {str(screenshot_error)}")
                            # Add text observation about the error instead
                            memory_step.observations = (
                                "Could not take screenshot. " 
                                if memory_step.observations is None 
                                else memory_step.observations + "\nCould not take screenshot. "
                            )
                            memory_step.observations += f"Current URL: {self.visual_tool.webdriver.current_url}"
                except Exception as e:
                    self.logger.error(f"Error saving screenshot to memory: {str(e)}")
                    # Ensure the step has text observations at minimum
                    if not hasattr(memory_step, 'observations') or memory_step.observations is None:
                        memory_step.observations = "Error capturing browser state."
            
            # Create the agent with step callbacks for screenshots
            self.agent = CodeAgent(
                tools=tools,
                model=model,
                additional_authorized_imports=["helium", "selenium", "time"] + self.authorized_imports,
                verbosity_level=2,
                step_callbacks=[save_screenshot_to_memory],
                max_steps=20  # Limit to 20 steps to avoid infinite loops
                # flatten_messages_as_text=False 
            )
            
            # Import helium for the agent
            # self.agent.python_executor("from helium import *", self.agent.state)
            self.agent.python_executor("from helium import *")
            # Start browser thread
            self.browser_thread.start()
            
            self.is_initialized = True
            self.logger.info(f"Visual web agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing visual web agent: {str(e)}")
            self.logger.error(traceback.format_exc())
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
        
        # Add additional tools from the notebook
        
        # Add search_item_ctrl_f tool
        @tool
        def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
            """
            Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
            Args:
                text: The text to search for
                nth_result: Which occurrence to jump to (default: 1)
            """
            driver = self.visual_tool.webdriver
            from selenium.webdriver.common.by import By
            
            elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
            if nth_result > len(elements):
                raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
            result = f"Found {len(elements)} matches for '{text}'."
            elem = elements[nth_result - 1]
            driver.execute_script("arguments[0].scrollIntoView(true);", elem)
            result += f" Focused on element {nth_result} of {len(elements)}"
            return result
        
        tools.append(search_item_ctrl_f)
        
        # Add go_back tool
        @tool
        def go_back() -> str:
            """Goes back to previous page."""
            if self.visual_tool.webdriver:
                self.visual_tool.webdriver.back()
                return "Navigated back to previous page"
            return "No browser session to navigate back"
        
        tools.append(go_back)
        
        # Add close_popups tool
        @tool
        def close_popups() -> str:
            """
            Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
            This does not work on cookie consent banners.
            """
            from selenium.webdriver.common.keys import Keys
            
            if self.visual_tool.webdriver:
                from selenium import webdriver
                webdriver.ActionChains(self.visual_tool.webdriver).send_keys(Keys.ESCAPE).perform()
                return "Attempted to close popups using Escape key"
            return "No browser session to close popups"
        
        tools.append(close_popups)
        
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
            # Add detailed instructions for helium
            helium_instructions = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
go_to('github.com/trending')
```<end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
click("Top products")
```<end_code>

If it's a link:
Code:
click(Link("Top products"))
```<end_code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
close_popups()
```<end_code>

You can use .exists() to check for the existence of an element. For example:
Code:
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
"""
            
            # Enhance the prompt with visual web automation guidance
            enhanced_prompt = f"""
You are a visual web automation agent that can control a web browser to accomplish tasks.
You can see screenshots of the browser window and interact with it using commands.

{helium_instructions}

You have access to these tools:
- web_search: Search the web for information
- search_item_ctrl_f: Search for text on the current page and jump to it
- close_popups: Close any popups using the Escape key
- go_back: Go back to the previous page
- take_screenshot: Take a screenshot and view it

USER TASK: {input_text}

First, take a screenshot to see the current state of the browser.
Then, complete the task step by step, making sure to check the results after each action by observing the screenshot.
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
        return ["web_search", "web_automation", "visual_interaction", "screenshot", "text_search"]
    
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