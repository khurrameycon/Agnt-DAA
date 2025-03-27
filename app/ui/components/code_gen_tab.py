"""
Code Generation Tab Component for SagaX1
Tab for code generation agent interaction
"""

import os
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QTabWidget,
    QCheckBox, QRadioButton, QGroupBox, QFormLayout
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
            pattern = f"\\b{word}\\b"
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
        self.highlighting_rules.append((r"\\b[0-9]+\\b", number_format))
        
        # Function format
        function_format = QTextCharFormat()
        function_format.setForeground(QColor(220, 220, 170))  # Yellow
        function_format.setFontWeight(QFont.Weight.Bold)
        
        # Add function rules
        self.highlighting_rules.append((r"\\b[A-Za-z0-9_]+(?=\\()", function_format))
        
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
            # Find all matches
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, format)


class CodeGenTab(QWidget):
    """Tab for code generation"""
    
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
        
        # Options button
        options_button = QPushButton("Options")
        options_button.clicked.connect(self.show_options)
        top_layout.addWidget(options_button)
        
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
        self.code_editor.setPlaceholderText("# Write your Python code here...")
        self.code_editor.setFont(QFont("Courier New", 10))
        
        # Add syntax highlighting
        self.highlighter = PythonSyntaxHighlighter(self.code_editor.document())
        
        code_layout.addWidget(QLabel("Code Editor:"))
        code_layout.addWidget(self.code_editor)
        
        # Create run button
        run_button = QPushButton("Run Code")
        run_button.clicked.connect(self.run_code)
        code_layout.addWidget(run_button)
        
        # Create output area
        self.code_output = QTextEdit()
        self.code_output.setReadOnly(True)
        self.code_output.setPlaceholderText("Code execution output will appear here...")
        self.code_output.setFont(QFont("Courier New", 10))
        
        code_layout.addWidget(QLabel("Output:"))
        code_layout.addWidget(self.code_output)
        
        # Add code editor tab
        self.tabs.addTab(code_tab, "Code Editor")
        
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
        """Create input panel for user commands"""
        input_layout = QHBoxLayout()
        
        # Command input
        self.command_input = QTextEdit()
        self.command_input.setPlaceholderText("Describe what code you want to generate...")
        self.command_input.setMaximumHeight(100)
        input_layout.addWidget(self.command_input)
        
        # Send button
        self.send_button = QPushButton("Generate Code")
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
        
        # Filter to code generation agents
        code_gen_agents = [
            agent for agent in active_agents
            if agent["agent_type"] == "code_generation"
        ]
        
        if not code_gen_agents:
            self.agent_selector.addItem("No code generation agents available")
            self.send_button.setEnabled(False)
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
            self.send_button.setEnabled(False)
            return
        
        try:
            # Update current agent
            self.current_agent_id = agent_id
            
            # Enable send button
            self.send_button.setEnabled(True)
        except Exception as e:
            self.logger.error(f"Error selecting agent: {str(e)}")
            self.current_agent_id = None
            self.send_button.setEnabled(False)
    
    def create_new_agent(self):
        """Create a new code generation agent"""
        # Signal to main window to open the create agent dialog with code_generation type
        self.parent().create_code_generation_agent()
    
    def show_options(self):
        """Show options dialog"""
        from PyQt6.QtWidgets import QDialog
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Code Generation Options")
        dialog.resize(400, 300)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Create options
        sandbox_group = QGroupBox("Code Execution")
        sandbox_layout = QVBoxLayout(sandbox_group)
        
        # Sandbox options
        self.sandbox_checkbox = QCheckBox("Run code in sandbox (safer but more limited)")
        self.sandbox_checkbox.setChecked(True)
        sandbox_layout.addWidget(self.sandbox_checkbox)
        
        # Add to layout
        layout.addWidget(sandbox_group)
        
        # Create import options
        import_group = QGroupBox("Authorized Imports")
        import_layout = QFormLayout(import_group)
        
        # Import options
        self.import_edit = QTextEdit()
        self.import_edit.setPlaceholderText("Enter comma-separated list of authorized imports...")
        
        # Get default imports from config
        default_imports = self.agent_manager.config_manager.get("execution.authorized_imports", [])
        self.import_edit.setText(", ".join(default_imports))
        import_layout.addRow("Imports:", self.import_edit)
        
        # Add to layout
        layout.addWidget(import_group)
        
        # Create button box
        from PyQt6.QtWidgets import QDialogButtonBox
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec():
            # Save options
            if self.current_agent_id:
                # Get agent config
                agent_config = self.agent_manager.get_agent_config(self.current_agent_id)
                
                # Update sandbox option
                agent_config["additional_config"]["sandbox"] = self.sandbox_checkbox.isChecked()
                
                # Update authorized imports
                import_text = self.import_edit.toPlainText()
                authorized_imports = [imp.strip() for imp in import_text.split(",") if imp.strip()]
                agent_config["additional_config"]["authorized_imports"] = authorized_imports
                
                # Update config
                self.agent_manager.agent_configs[self.current_agent_id] = agent_config
    
    def run_code(self):
        """Run the code in the editor"""
        if self.current_agent_id is None:
            return
        
        # Get code
        code = self.code_editor.toPlainText().strip()
        if not code:
            return
        
        # Clear output
        self.code_output.clear()
        
        # Get agent
        agent = self.agent_manager.active_agents.get(self.current_agent_id)
        
        if agent is None:
            # Create agent if it doesn't exist
            self.agent_manager.create_agent(
                agent_id=self.current_agent_id,
                agent_type="code_generation",
                model_config=self.agent_manager.get_agent_config(self.current_agent_id)["model_config"],
                tools=self.agent_manager.get_agent_config(self.current_agent_id)["tools"],
                additional_config=self.agent_manager.get_agent_config(self.current_agent_id)["additional_config"]
            )
            
            agent = self.agent_manager.active_agents.get(self.current_agent_id)
            
            if agent is None:
                self.logger.error(f"Failed to create agent {self.current_agent_id}")
                self.code_output.setText("Error: Failed to create agent")
                return
        
        # Initialize agent
        if not getattr(agent, "is_initialized", False):
            agent.initialize()
        
        # Run code
        try:
            # Get python execution tool
            python_tool = agent._initialize_tools()[0]
            
            # Run code
            result = python_tool(code)
            
            # Show result
            self.code_output.setText(result)
        except Exception as e:
            self.code_output.setText(f"Error: {str(e)}")
    
    def send_command(self):
        """Send command to agent"""
        if self.current_agent_id is None:
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
        
        # Switch to conversation tab
        self.tabs.setCurrentIndex(1)
        
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
        
        # Parse code blocks from result
        import re
        code_blocks = re.findall(r"```python\n(.*?)```", result, re.DOTALL)
        
        if code_blocks:
            # Get first code block
            code = code_blocks[0].strip()
            
            # Set code in editor
            self.code_editor.setText(code)
            
            # Switch to code editor tab
            self.tabs.setCurrentIndex(0)
        
        # Enable input
        self.command_input.setEnabled(True)
        self.send_button.setEnabled(True)