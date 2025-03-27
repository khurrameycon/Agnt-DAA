"""
Code Generation Agent for SagaX1
Agent that can generate and execute code to solve problems
"""

import os
import sys
import logging
import tempfile
import subprocess
import traceback
from typing import Dict, Any, List, Optional, Callable

from app.agents.base_agent import BaseAgent
from smolagents import CodeAgent, tool

class PythonExecutionTool(tool):
    """Tool for executing Python code"""
    
    def __init__(self, sandbox=True):
        """Initialize the Python execution tool
        
        Args:
            sandbox: Whether to run in a sandboxed environment
        """
        self.sandbox = sandbox
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, code: str) -> str:
        """Execute Python code and return the result
        
        Args:
            code: Python code to execute
            
        Returns:
            Output of code execution
        """
        if self.sandbox:
            return self._run_in_sandbox(code)
        else:
            return self._run_locally(code)
    
    def _run_locally(self, code: str) -> str:
        """Run code locally using exec()
        
        Args:
            code: Python code to execute
            
        Returns:
            Output of code execution
        """
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Create output buffers
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # Create local scope
        local_scope = {}
        
        try:
            # Execute code with redirected output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, globals(), local_scope)
            
            # Get output
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()
            
            # Return result
            if stderr_output:
                result = f"Code execution completed with errors:\n\n{stderr_output}\n\nStandard output:\n{stdout_output}"
            else:
                result = f"Code execution completed successfully:\n\n{stdout_output}"
            
            return result
            
        except Exception as e:
            # Get traceback
            tb = traceback.format_exc()
            
            # Get any output before the error
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()
            
            # Return error
            return f"Code execution failed with error:\n\n{tb}\n\nStandard output:\n{stdout_output}\n\nStandard error:\n{stderr_output}"
    
    def _run_in_sandbox(self, code: str) -> str:
        """Run code in a sandbox (separate process)
        
        Args:
            code: Python code to execute
            
        Returns:
            Output of code execution
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run code in a separate process
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Get output
            stdout_output = result.stdout
            stderr_output = result.stderr
            
            # Return result
            if result.returncode != 0:
                output = f"Code execution failed with error code {result.returncode}:\n\n{stderr_output}\n\nStandard output:\n{stdout_output}"
            else:
                output = f"Code execution completed successfully:\n\n{stdout_output}"
            
            return output
            
        except subprocess.TimeoutExpired:
            return "Code execution timed out after 30 seconds"
        except Exception as e:
            return f"Error running code: {str(e)}"
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

class CodeGenerationAgent(BaseAgent):
    """Agent for generating and executing code"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the code generation agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                max_tokens: Maximum number of tokens to generate
                temperature: Temperature for generation
                sandbox: Whether to run code in a sandbox
        """
        super().__init__(agent_id, config)
        
        self.model_id = config.get("model_id", "meta-llama/Llama-3-8B-Instruct")
        self.device = config.get("device", "auto")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        self.authorized_imports = config.get("authorized_imports", [])
        self.sandbox = config.get("sandbox", True)
        
        self.agent = None
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the model and agent"""
        if self.is_initialized:
            return
        
        try:
            from smolagents import TransformersModel, HfApiModel, OpenAIServerModel, LiteLLMModel
            
            self.logger.info(f"Initializing code generation agent with model {self.model_id}")
            
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
                additional_authorized_imports=self.authorized_imports,
                verbosity_level=1
            )
            
            self.is_initialized = True
            self.logger.info(f"Code generation agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing code generation agent: {str(e)}")
            raise
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize tools for the agent
        
        Returns:
            List of tools
        """
        tools = []
        
        # Add Python execution tool
        tools.append(PythonExecutionTool(sandbox=self.sandbox))
        
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
            # Enhance the prompt with code generation guidance
            enhanced_prompt = f"""
You are a code generation agent that can write and execute Python code to solve problems.
You have the ability to:
1. Generate Python code to solve a given problem
2. Execute the code and see the results
3. Refine the code based on the execution results

When writing code, follow these best practices:
- Include appropriate comments to explain your code
- Handle potential errors with try-except blocks
- Break down complex problems into smaller functions
- Use clear variable and function names

USER PROBLEM: {input_text}

First, understand the problem clearly. Then write Python code to solve it, and execute it to verify your solution.
"""
            
            # Run the agent
            result = self.agent.run(enhanced_prompt)
            
            # Add to history
            self.add_to_history(input_text, str(result))
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Error running code generation agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while generating code: {error_msg}"
    
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
        return ["code_generation", "code_execution", "problem_solving"]