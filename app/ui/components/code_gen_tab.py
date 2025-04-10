"""
Code Generation Tab Component for sagax1
Tab for code generation agent interaction with Hugging Face spaces
"""

import os
import logging
import re
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QTabWidget,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QFont, QColor

from app.core.agent_manager import AgentManager
from app.ui.components.conversation import ConversationWidget


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code"""
    
    def __init__(self, parent=None):
        """Initialize the syntax highlighter
        
        Args:
            parent: Parent text document
        """
        super().__init__(parent)
        
        self.highlighting_rules = []
        
        # Keyword format
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(86, 156, 214))  # Blue
        keyword_format.setFontWeight(QFont.Weight.Bold)
        
        # Keywords
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "exec", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda",
            "not", "or", "pass", "print", "raise", "return", "try",
            "while", "with", "yield", "None", "True", "False"
        ]
        
        # Add keyword rules
        for word in keywords:
            pattern = r"\b" + word + r"\b"  # Fixed regex pattern
            self.highlighting_rules.append((pattern, keyword_format))
        
        # String format
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(214, 157, 133))  # Orange
        
        # Add string rules
        self.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"', string_format))
        self.highlighting_rules.append((r"'[^'\\]*(\\.[^'\\]*)*'", string_format))
        
        # Number format
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(181, 206, 168))  # Light green
        
        # Add number rules
        self.highlighting_rules.append((r"\b[0-9]+\b", number_format))
        
        # Function format
        function_format = QTextCharFormat()
        function_format.setForeground(QColor(220, 220, 170))  # Yellow
        function_format.setFontWeight(QFont.Weight.Bold)
        
        # Add function rules
        self.highlighting_rules.append((r"\b[A-Za-z0-9_]+(?=\()", function_format))
        
        # Comment format
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(87, 166, 74))  # Green
        
        # Add comment rules
        self.highlighting_rules.append((r"#[^\n]*", comment_format))
    
    def highlightBlock(self, text):
        """Highlight a block of text
        
        Args:
            text: Text to highlight
        """
        import re
        
        for pattern, format in self.highlighting_rules:
            # Find all matches with error handling
            try:
                for match in re.finditer(pattern, text):
                    start, end = match.span()
                    self.setFormat(start, end - start, format)
            except Exception as e:
                # Skip this pattern if there's an error
                continue


class CodeGenTab(QWidget):
    """Tab for code generation using Hugging Face spaces"""
    
    def __init__(self, agent_manager: AgentManager, parent=None):
        """Initialize the code generation tab
        
        Args:
            agent_manager: Agent manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.agent_manager = agent_manager
        self.logger = logging.getLogger(__name__)
        self.current_agent_id = None
        
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
        top_layout.addWidget(QLabel("Select Code Generation Agent:"))
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
        
        top_layout.addStretch()
        
        self.layout.addLayout(top_layout)
    
    def create_main_panel(self):
        """Create main panel with code editor, output, and conversation"""
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create code editor tab
        code_tab = QWidget()
        code_layout = QVBoxLayout(code_tab)
        
        # Create code editor
        self.code_editor = QTextEdit()
        self.code_editor.setPlaceholderText("# Generated code will appear here...")
        self.code_editor.setFont(QFont("Courier New", 10))
        
        # Add syntax highlighting
        self.highlighter = PythonSyntaxHighlighter(self.code_editor.document())
        
        code_layout.addWidget(QLabel("Generated Code:"))
        code_layout.addWidget(self.code_editor)
        
        # Create copy and save buttons
        button_layout = QHBoxLayout()
        
        copy_button = QPushButton("Copy Code")
        copy_button.clicked.connect(self.copy_code)
        button_layout.addWidget(copy_button)
        
        save_button = QPushButton("Save Code")
        save_button.clicked.connect(self.save_code)
        button_layout.addWidget(save_button)
        
        button_layout.addStretch()
        code_layout.addLayout(button_layout)
        
        # Add code editor tab
        self.tabs.addTab(code_tab, "Generated Code")
        
        # Create conversation tab
        conversation_tab = QWidget()
        conversation_layout = QVBoxLayout(conversation_tab)
        
        # Create conversation widget
        self.conversation = ConversationWidget()
        conversation_layout.addWidget(self.conversation)
        
        # Add conversation tab
        self.tabs.addTab(conversation_tab, "Conversation")
        
        # Add tabs to layout
        self.layout.addWidget(self.tabs, stretch=1)
    
    def create_input_panel(self):
        """Create input panel for user prompts"""
        input_layout = QHBoxLayout()
        
        # Prompt input
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt for code generation here...\nFor example: 'Create a Python function to sort a list of dictionaries by a specific key'")
        self.prompt_input.setMinimumHeight(100)
        input_layout.addWidget(self.prompt_input)
        
        # Generate button
        self.generate_button = QPushButton("Generate Code")
        self.generate_button.clicked.connect(self.generate_code)
        self.generate_button.setEnabled(False)
        input_layout.addWidget(self.generate_button)
        
        self.layout.addLayout(input_layout)
    
    def load_agents(self):
        """Load available agents"""
        # Clear existing items
        self.agent_selector.clear()
        
        # Get active agents
        active_agents = self.agent_manager.get_active_agents()
        
        # Filter to code generation agents
        code_gen_agents = [
            agent for agent in active_agents
            if agent["agent_type"] == "code_generation"
        ]
        
        if not code_gen_agents:
            self.agent_selector.addItem("No code generation agents available")
            self.generate_button.setEnabled(False)
            return
        
        # Add to combo box
        for agent in code_gen_agents:
            self.agent_selector.addItem(agent["agent_id"])
        
        # Select first agent
        if self.agent_selector.count() > 0:
            self.agent_selector.setCurrentIndex(0)
    
    def on_agent_selected(self, agent_id: str):
        """Handle agent selection
        
        Args:
            agent_id: ID of the selected agent
        """
        if agent_id == "No code generation agents available":
            self.current_agent_id = None
            self.generate_button.setEnabled(False)
            return
        
        try:
            # Update current agent
            self.current_agent_id = agent_id
            
            # Enable generate button
            self.generate_button.setEnabled(True)
        except Exception as e:
            self.logger.error(f"Error selecting agent: {str(e)}")
            self.current_agent_id = None
            self.generate_button.setEnabled(False)
    
    def create_new_agent(self):
        """Create a new code generation agent"""
        # Get the main window
        main_window = self.window()
        
        # Check if main_window has the method
        if hasattr(main_window, 'create_code_generation_agent'):
            main_window.create_code_generation_agent()
        else:
            # Fallback if we can't find the method
            QMessageBox.warning(
                self,
                "Not Implemented",
                "Create code generation agent functionality not found in main window."
            )
    
    def copy_code(self):
        """Copy the generated code to clipboard"""
        from PyQt6.QtWidgets import QApplication
        
        # Get the code
        code = self.code_editor.toPlainText()
        if not code:
            return
        
        # Copy to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(code)
        
        # Show success message
        main_window = self.window()
        if hasattr(main_window, 'status_bar'):
            main_window.status_bar.showMessage("Code copied to clipboard", 3000)
    
    def save_code(self):
        """Save the generated code to a file"""
        from PyQt6.QtWidgets import QFileDialog
        
        # Get the code
        code = self.code_editor.toPlainText()
        if not code:
            return
        
        # Guess the file extension
        extension = ".py"  # Default to Python
        if "function" in code and "{" in code:
            extension = ".js"
        elif "public class" in code or "public static void" in code:
            extension = ".java"
        elif "#include" in code:
            extension = ".cpp"
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Code",
            "",
            f"Code Files (*{extension});;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save code to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Show success message
            main_window = self.window()
            if hasattr(main_window, 'status_bar'):
                main_window.status_bar.showMessage(f"Code saved to {file_path}", 3000)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Save Failed",
                f"Failed to save code: {str(e)}"
            )
    
    def generate_code(self):
        """Generate code from prompt using AI"""
        if self.current_agent_id is None:
            return
        
        # Get prompt
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            return
        
        # Add to conversation
        self.conversation.add_message(prompt, is_user=True)
        
        # Clear input
        self.prompt_input.clear()
        
        # Disable input
        self.prompt_input.setEnabled(False)
        self.generate_button.setEnabled(False)
        
        # Update status
        main_window = self.window()
        if hasattr(main_window, 'status_bar'):
            main_window.status_bar.showMessage("Generating code...", 0)
        
        # Run agent in thread
        main_window = self.window()
        if hasattr(main_window, 'run_agent_in_thread'):
            main_window.run_agent_in_thread(
                self.current_agent_id, 
                prompt,
                self.handle_agent_result
            )
        else:
            # Enable input again if we can't find the method
            self.prompt_input.setEnabled(True)
            self.generate_button.setEnabled(True)
            self.conversation.add_message("Error: Unable to run agent thread", is_user=False)
    
    def handle_agent_result(self, result: str):
        """Handle agent result
        
        Args:
            result: Agent result
        """
        # Add to conversation
        self.conversation.add_message(result, is_user=False)
        
        # Extract code blocks from result
        code = self.extract_code_from_result(result)
        
        # Set code in editor
        self.code_editor.setText(code)
        
        # Switch to code tab
        self.tabs.setCurrentIndex(0)
        
        # Enable input
        self.prompt_input.setEnabled(True)
        self.generate_button.setEnabled(True)
        
        # Update status
        main_window = self.window()
        if hasattr(main_window, 'status_bar'):
            main_window.status_bar.showMessage("Code generation complete", 3000)
    
    def extract_code_from_result(self, result: str) -> str:
        """Extract code blocks from the result
        
        Args:
            result: Result from the agent
            
        Returns:
            Extracted code
        """
        # Extract code from markdown code blocks
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", result, re.DOTALL)
        
        if code_blocks:
            # Return the first code block
            return code_blocks[0].strip()
        
        # If no code blocks found, just return the result
        return result