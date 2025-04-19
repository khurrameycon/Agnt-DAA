"""
Main Window for sagax1
Main application window and UI logic
"""


import sys
import logging
from typing import Callable
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QComboBox,
    QStatusBar, QTabWidget, QLineEdit, QMessageBox,
    QMenuBar, QMenu, QDialog, QDialogButtonBox, QFormLayout,
    QListWidget, QListWidgetItem, QSplitter, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QThread, pyqtSlot
from PyQt6.QtGui import QIcon, QAction, QFont
from PyQt6.QtWidgets import QTextBrowser
from app.core.config_manager import ConfigManager
from app.core.agent_manager import AgentManager
from app.ui.dialogs.create_agent_dialog import CreateAgentDialog
from app.ui.components.conversation import ConversationWidget
from app.ui.components.fine_tuning_tab import FineTuningTab
import markdown 
from app.agents.rag_agent import RagAgent

class AgentThread(QThread):
    """Thread for running agents to keep UI responsive"""
    
    result_ready = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, agent_manager, agent_id, input_text):
        """Initialize the agent thread
        
        Args:
            agent_manager: Agent manager instance
            agent_id: ID of the agent to run
            input_text: Input text for the agent
        """
        super().__init__()
        self.agent_manager = agent_manager
        self.agent_id = agent_id
        self.input_text = input_text
        
    
    def run(self):
        """Run the agent"""
        def progress_callback(text):
            self.progress_update.emit(text)
        
        # Run the agent
        result = self.agent_manager.run_agent(
            self.agent_id, 
            self.input_text,
            callback=progress_callback
        )
        
        # Emit the result
        self.result_ready.emit(result)

class ApiKeyDialog(QDialog):
    """Dialog for entering API key"""
    
    def __init__(self, parent=None, current_key=""):
        super().__init__(parent)
        self.setWindowTitle("Enter Hugging Face API Key")
        self.resize(400, 150)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form
        form_layout = QFormLayout()
        self.api_key_input = QLineEdit(current_key)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        form_layout.addRow("API Key:", self.api_key_input)
        
        # Add info text
        info_label = QLabel(
            "You can get your Hugging Face API key from "
            "https://huggingface.co/settings/tokens"
        )
        info_label.setWordWrap(True)
        
        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Add to layout
        layout.addLayout(form_layout)
        layout.addWidget(info_label)
        layout.addWidget(button_box)
    
    def get_api_key(self):
        """Get the entered API key"""
        return self.api_key_input.text()

class MainWindow(QMainWindow):
    """Main window for the sagax1 application"""
    
    def __init__(self, agent_manager: AgentManager, config_manager: ConfigManager):
        """Initialize the main window
        
        Args:
            agent_manager: Agent manager instance
            config_manager: Configuration manager instance
        """
        super().__init__()
        
        self.agent_manager = agent_manager
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("sagax1 - AI Agent Platform")
        self.resize(1024, 768)
        
        # Create main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create menu
        self.create_menu()
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_chat_tab()
        self.create_web_browsing_tab()
        self.create_visual_web_tab()
        self.create_code_gen_tab()
        self.create_media_gen_tab()
        self.create_fine_tuning_tab()
        self.create_rag_tab()
        self.create_agents_tab()
        self.create_settings_tab()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Load agents
        self.load_agents()
        
        # Check for API key
        self.check_api_key()

        # After creating the visual_web_tab
       
        
    def check_api_key(self):
        """Check if API key is set and prompt if not"""
        api_key = self.config_manager.get_hf_api_key()
        if not api_key:
            QMessageBox.information(
                self,
                "API Key Required",
                "Please set your Hugging Face API key to enable full functionality."
            )
            self.set_api_key()
    
    def set_api_key(self):
        """Set the Hugging Face API key"""
        current_key = self.config_manager.get_hf_api_key() or ""
        dialog = ApiKeyDialog(self, current_key)
        
        if dialog.exec():
            api_key = dialog.get_api_key()
            self.config_manager.set_hf_api_key(api_key)
            self.status_bar.showMessage("API key updated", 3000)
            
            # Update the field in settings tab
            if hasattr(self, "api_key_field"):
                self.api_key_field.setText(api_key)
    
    def create_menu(self):
        """Create the application menu"""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        
        # File menu
        file_menu = QMenu("&File", self)
        menu_bar.addMenu(file_menu)
        
        # Add actions to file menu
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Agent menu
        agent_menu = QMenu("&Agent", self)
        menu_bar.addMenu(agent_menu)
        
        create_agent_action = QAction("&Create New Agent", self)
        create_agent_action.triggered.connect(self.create_new_agent)
        agent_menu.addAction(create_agent_action)
        
        # Settings menu
        settings_menu = QMenu("&Settings", self)
        menu_bar.addMenu(settings_menu)
        
        api_key_action = QAction("Set &API Key", self)
        api_key_action.triggered.connect(self.set_api_key)
        settings_menu.addAction(api_key_action)
        
        # Help menu
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    

    def convert_markdown_to_html(self, markdown_text):
        """Convert markdown text to HTML for display
        
        Args:
            markdown_text: Markdown formatted text
            
        Returns:
            HTML formatted text
        """
        try:
            # Use the markdown library to convert to HTML
            html = markdown.markdown(markdown_text, extensions=['tables', 'fenced_code'])
            
            # Add some basic styling
            styled_html = f"""
            <html>
            <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; font-size: 24px; margin-top: 20px; margin-bottom: 10px; }}
                h2 {{ color: #3498db; font-size: 20px; margin-top: 15px; margin-bottom: 10px; }}
                h3 {{ color: #2980b9; font-size: 16px; margin-top: 10px; margin-bottom: 5px; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                pre {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                code {{ font-family: Consolas, monospace; }}
                blockquote {{ border-left: 4px solid #3498db; padding-left: 10px; color: #7f8c8d; margin-left: 20px; }}
                ul, ol {{ padding-left: 30px; }}
                li {{ margin-bottom: 5px; }}
                p {{ margin-bottom: 10px; line-height: 1.5; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 15px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
            </head>
            <body>
            {html}
            </body>
            </html>
            """
            return styled_html
        except Exception as e:
            # Fallback if markdown conversion fails
            self.logger.error(f"Error converting markdown to HTML: {str(e)}")
            return f"<pre>{markdown_text}</pre>"
    
    def create_chat_tab(self):
        """Create the chat tab with enhanced support for execution mode display"""
        chat_tab = QWidget()
        layout = QVBoxLayout(chat_tab)
        
        # Create a splitter for agent selector and conversation
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)
        
        # Top widget for agent selection
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # Agent selection
        agent_layout = QHBoxLayout()
        agent_layout.addWidget(QLabel("Select Agent:"))
        self.agent_selector = QComboBox()
        agent_layout.addWidget(self.agent_selector)
        
        # Add buttons
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_agents)
        agent_layout.addWidget(refresh_button)
        
        create_button = QPushButton("Create New")
        create_button.clicked.connect(self.create_new_agent)
        agent_layout.addWidget(create_button)
        
        # Add execution mode indicator label
        self.execution_mode_label = QLabel("Mode: Not Selected")
        self.execution_mode_label.setStyleSheet("font-style: italic; color: #666;")
        agent_layout.addWidget(self.execution_mode_label)
        
        agent_layout.addStretch()
        top_layout.addLayout(agent_layout)
        
        # Add agent info
        self.agent_info = QLabel("No agent selected")
        top_layout.addWidget(self.agent_info)
        
        splitter.addWidget(top_widget)
        
        # Conversation area
        conversation_widget = QWidget()
        conversation_layout = QVBoxLayout(conversation_widget)
        
        # Conversation history
        self.conversation = ConversationWidget()
        conversation_layout.addWidget(self.conversation)
        
        # Input area
        input_layout = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Type your message here...")
        self.chat_input.setMaximumHeight(100)
        input_layout.addWidget(self.chat_input)
        
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)
        conversation_layout.addLayout(input_layout)
        
        splitter.addWidget(conversation_widget)
        
        # Connect signals
        self.agent_selector.currentTextChanged.connect(self.on_agent_selected)
        
        self.tabs.addTab(chat_tab, "Chat")
    
    def create_agents_tab(self):
        """Create the agents tab"""
        agents_tab = QWidget()
        layout = QVBoxLayout(agents_tab)
        
        # Agent list
        layout.addWidget(QLabel("Available Agents:"))
        self.agent_list = QListWidget()
        layout.addWidget(self.agent_list)
        
        # Agent details
        self.agent_details = QTextEdit()
        self.agent_details.setReadOnly(True)
        layout.addWidget(QLabel("Agent Details:"))
        layout.addWidget(self.agent_details)
        
        # Buttons
        button_layout = QHBoxLayout()
        create_button = QPushButton("Create New Agent")
        create_button.clicked.connect(self.create_new_agent)
        button_layout.addWidget(create_button)
        
        self.delete_button = QPushButton("Delete Selected Agent")
        self.delete_button.clicked.connect(self.delete_selected_agent)
        self.delete_button.setEnabled(False)
        button_layout.addWidget(self.delete_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Connect signals
        self.agent_list.itemSelectionChanged.connect(self.on_agent_list_selection_changed)
        
        self.tabs.addTab(agents_tab, "Agents")
    
    def create_settings_tab(self):
        """Create the settings tab"""
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)
        
        # API settings
        layout.addWidget(QLabel("Hugging Face API Settings:"))
        api_layout = QFormLayout()
        
        # API key field (masked)
        self.api_key_field = QLineEdit()
        self.api_key_field.setEchoMode(QLineEdit.EchoMode.Password)
        api_key = self.config_manager.get_hf_api_key() or ""
        self.api_key_field.setText(api_key)
        self.api_key_field.setReadOnly(True)
        
        api_layout.addRow("API Key:", self.api_key_field)
        
        # API key change button
        change_key_button = QPushButton("Change")
        change_key_button.clicked.connect(self.set_api_key)
        api_layout.addRow("", change_key_button)
        
        layout.addLayout(api_layout)
        
        # Model cache settings
        layout.addWidget(QLabel("Model Cache Settings:"))
        cache_layout = QFormLayout()
        
        # Cache directory
        self.cache_dir_field = QLineEdit()
        self.cache_dir_field.setText(self.config_manager.get("models.cache_dir", "~/.cache/sagax1/models"))
        self.cache_dir_field.setReadOnly(True)
        cache_layout.addRow("Cache Directory:", self.cache_dir_field)
        
        # Cache size
        self.cache_size_label = QLabel("Calculating...")
        cache_layout.addRow("Cache Size:", self.cache_size_label)
        
        # Clear cache button
        clear_cache_button = QPushButton("Clear Cache")
        clear_cache_button.clicked.connect(self.clear_model_cache)
        cache_layout.addRow("", clear_cache_button)
        
        layout.addLayout(cache_layout)
        layout.addStretch()
        
        self.tabs.addTab(settings_tab, "Settings")
        
        # Start a thread to calculate cache size
        self.update_cache_size()
    
    def load_agents(self):
        """Load available agents"""
        # Clear existing items
        self.agent_selector.clear()
        self.agent_list.clear()
        
        # Get active agents
        active_agents = self.agent_manager.get_active_agents()
        
        if not active_agents:
            self.agent_selector.addItem("No agents available")
            return
        
        # Add to combo box and list
        for agent in active_agents:
            agent_id = agent["agent_id"]
            self.agent_selector.addItem(agent_id)
            
            # Add to list widget
            item = QListWidgetItem(agent_id)
            item.setData(Qt.ItemDataRole.UserRole, agent)
            self.agent_list.addItem(item)
    
    def on_agent_selected(self, agent_id: str):
        """Handle agent selection with enhanced execution mode display
        
        Args:
            agent_id: ID of the selected agent
        """
        if not agent_id or agent_id == "No agents available":
            self.agent_info.setText("No agent selected")
            self.execution_mode_label.setText("Mode: Not Selected")
            return
        
        try:
            # Get agent config
            agent_config = self.agent_manager.get_agent_config(agent_id)
            
            # Update agent info
            model_id = agent_config["model_config"].get("model_id", "unknown")
            agent_type = agent_config["agent_type"]
            tools = ", ".join(agent_config["tools"]) if agent_config["tools"] else "None"
            
            info_text = f"Agent: {agent_id}\nType: {agent_type}\nModel: {model_id}\nTools: {tools}"
            self.agent_info.setText(info_text)
            
            # Update execution mode label
            use_api = agent_config["model_config"].get("use_api", False)
            if use_api:
                self.execution_mode_label.setText("Mode: Inference API")
                self.execution_mode_label.setStyleSheet("color: #2E86C1; font-weight: bold;")
            else:
                self.execution_mode_label.setText("Mode: Local Model")
                self.execution_mode_label.setStyleSheet("color: #27AE60; font-weight: bold;")
            
        except Exception as e:
            self.logger.error(f"Error getting agent config: {str(e)}")
            self.agent_info.setText(f"Error: {str(e)}")
            self.execution_mode_label.setText("Mode: Unknown")
            self.execution_mode_label.setStyleSheet("color: #E74C3C; font-style: italic;")
    
    def on_agent_list_selection_changed(self):
        """Handle agent list selection change"""
        selected_items = self.agent_list.selectedItems()
        
        if selected_items:
            # Enable delete button
            self.delete_button.setEnabled(True)
            
            # Get agent data
            agent_data = selected_items[0].data(Qt.ItemDataRole.UserRole)
            
            # Show agent details
            details = f"Agent ID: {agent_data['agent_id']}\n"
            details += f"Agent Type: {agent_data['agent_type']}\n"
            details += f"Model: {agent_data['model_id']}\n"
            details += f"Tools: {', '.join(agent_data['tools'])}\n"
            
            self.agent_details.setText(details)
        else:
            # Disable delete button
            self.delete_button.setEnabled(False)
            
            # Clear agent details
            self.agent_details.clear()
    
    def delete_selected_agent(self):
        """Delete the selected agent"""
        selected_items = self.agent_list.selectedItems()
        
        if not selected_items:
            return
        
        # Get agent ID
        agent_data = selected_items[0].data(Qt.ItemDataRole.UserRole)
        agent_id = agent_data["agent_id"]
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete agent '{agent_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Delete agent
            self.agent_manager.remove_agent(agent_id)
            
            # Reload agents
            self.load_agents()
            
            # Show message
            self.status_bar.showMessage(f"Agent '{agent_id}' deleted", 3000)
    
    def send_message(self):
        """Send a message to the selected agent"""
        message = self.chat_input.toPlainText().strip()
        if not message:
            return
        
        # Get selected agent
        agent_id = self.agent_selector.currentText()
        if agent_id == "No agents available":
            QMessageBox.warning(
                self,
                "No Agent Selected",
                "Please create or select an agent first."
            )
            return
        
        # Add user message to chat history
        self.conversation.add_message(message, is_user=True)
        
        # Clear input field
        self.chat_input.clear()
        
        # Disable input during processing
        self.chat_input.setEnabled(False)
        
        # Show processing message
        self.status_bar.showMessage(f"Processing with agent {agent_id}...")
        
        # Create and start agent thread
        self.agent_thread = AgentThread(self.agent_manager, agent_id, message)
        self.agent_thread.result_ready.connect(self.handle_agent_result)
        self.agent_thread.progress_update.connect(self.handle_agent_progress)
        self.agent_thread.start()
    
    def handle_agent_result(self, result):
        """Handle agent result
        
        Args:
            result: Agent result
        """
        # Add agent message to chat history
        self.conversation.add_message(result, is_user=False)
        
        # Enable input
        self.chat_input.setEnabled(True)
        
        # Update status
        self.status_bar.showMessage("Ready", 3000)
    
    def handle_agent_progress(self, progress):
        """Handle agent progress update
        
        Args:
            progress: Progress update
        """
        # Update status bar
        self.status_bar.showMessage(f"Processing: {progress}")
    
    def create_new_agent(self):
        """Create a new agent"""
        dialog = CreateAgentDialog(self.agent_manager, self)
        
        if dialog.exec():
            # Get agent configuration
            agent_config = dialog.get_agent_config()
            
            # Create agent
            agent_id = self.agent_manager.create_agent(
                agent_id=agent_config["agent_id"],
                agent_type=agent_config["agent_type"],
                model_config=agent_config["model_config"],
                tools=agent_config["tools"],
                additional_config=agent_config["additional_config"]
            )
            
            if agent_id:
                # Show success message
                self.status_bar.showMessage(f"Agent '{agent_id}' created", 3000)
                
                # Set as default if requested
                if agent_config["additional_config"].get("is_default", False):
                    self.config_manager.set("agents.default_agent", agent_id)
                
                # Reload agents
                self.load_agents()
                
                # Select the new agent
                index = self.agent_selector.findText(agent_id)
                if index >= 0:
                    self.agent_selector.setCurrentIndex(index)
            else:
                QMessageBox.warning(
                    self,
                    "Agent Creation Failed",
                    "Failed to create agent. Check the logs for details."
                )
    
    def update_cache_size(self):
        """Update the cache size label"""
        import os
        import threading
        
        def calculate_size():
            # Get cache directory
            cache_dir = os.path.expanduser(self.cache_dir_field.text())
            
            if not os.path.exists(cache_dir):
                size_str = "0 bytes"
            else:
                total_size = 0
                for dirpath, _, filenames in os.walk(cache_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                
                # Convert to human-readable format
                for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
                    if total_size < 1024.0:
                        size_str = f"{total_size:.1f} {unit}"
                        break
                    total_size /= 1024.0
            
            # Update label in main thread
            self.cache_size_label.setText(size_str)
        
        # Start thread
        threading.Thread(target=calculate_size).start()
    
    def clear_model_cache(self):
        """Clear the model cache"""
        # Confirm
        confirm = QMessageBox.question(
            self,
            "Confirm Cache Clear",
            "Are you sure you want to clear the model cache? This will delete all downloaded models.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            import shutil
            import os
            
            # Get cache directory
            cache_dir = os.path.expanduser(self.cache_dir_field.text())
            
            if os.path.exists(cache_dir):
                try:
                    # Delete cache directory
                    shutil.rmtree(cache_dir)
                    
                    # Recreate empty directory
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    # Update cache size
                    self.update_cache_size()
                    
                    # Show success message
                    self.status_bar.showMessage("Cache cleared", 3000)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Cache Clear Failed",
                        f"Failed to clear cache: {str(e)}"
                    )
    
    def show_about(self):
        """Show the about dialog"""
        QMessageBox.about(
            self,
            "About sagax1",
            "<h1>sagax1</h1>"
            "<p>Version 0.1.0</p>"
            "<p>An Opensource AI-powered agent platform for everyday tasks</p>"
            "<p>&copy; 2025 sagax1 Team</p>"
        )


    def on_web_agent_selected(self, index):
        """Handle web agent selection
        
        Args:
            index: Index of the selected agent
        """
        if index < 0 or self.web_agent_selector.currentText() == "No web browsing agents available":
            self.multi_agent_toggle.setEnabled(False)
            self.web_send_button.setEnabled(False)
            return
        
        # Get the selected agent ID from item data
        agent_id = self.web_agent_selector.itemData(index)
        
        if not agent_id:
            return
        
        try:
            # Get agent config
            agent_config = self.agent_manager.get_agent_config(agent_id)
            
            # Update multi-agent toggle
            is_multi_agent = agent_config["additional_config"].get("multi_agent", False)
            self.multi_agent_toggle.setChecked(is_multi_agent)
            self.multi_agent_toggle.setEnabled(True)
            
            # Enable send button
            self.web_send_button.setEnabled(True)
        except Exception as e:
            self.logger.error(f"Error getting web agent config: {str(e)}")
            self.multi_agent_toggle.setEnabled(False)

    def toggle_multi_agent_mode(self, state):
        """Toggle multi-agent mode for web browsing agent
        
        Args:
            state: Checkbox state
        """
        if self.web_agent_selector.currentText() == "No web browsing agents available":
            return
        
        # Get the selected agent ID from item data
        agent_id = self.web_agent_selector.itemData(self.web_agent_selector.currentIndex())
        
        if not agent_id:
            return
        
        try:
            # Get agent config
            agent_config = self.agent_manager.get_agent_config(agent_id)
            
            # Update multi-agent setting
            agent_config["additional_config"]["multi_agent"] = bool(state)
            
            # Update agent config
            self.agent_manager.agent_configs[agent_id] = agent_config
            
            # If agent is active, recreate it
            if agent_id in self.agent_manager.active_agents:
                self.agent_manager.create_agent(
                    agent_id=agent_id,
                    agent_type="web_browsing",
                    model_config=agent_config["model_config"],
                    tools=agent_config["tools"],
                    additional_config=agent_config["additional_config"]
                )
            
            # Update display name in selector
            current_index = self.web_agent_selector.currentIndex()
            display_name = f"{agent_id}{' (Multi-Agent)' if bool(state) else ''}"
            self.web_agent_selector.setItemText(current_index, display_name)
            
            # Show status message
            mode_name = "Multi-Agent" if bool(state) else "Single-Agent"
            self.status_bar.showMessage(f"Web Browsing Agent switched to {mode_name} mode", 3000)
        except Exception as e:
            self.logger.error(f"Error updating web agent config: {str(e)}")

    def save_web_content(self):
        """Save web content to a file"""
        if not self.web_content_display.toPlainText():
            self.status_bar.showMessage("No web content to save", 3000)
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Web Content",
            "",
            "Text Files (*.txt);;HTML Files (*.html);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.web_content_display.toPlainText())
            
            self.status_bar.showMessage(f"Web content saved to {file_path}", 3000)
        except Exception as e:
            self.logger.error(f"Error saving web content: {str(e)}")
            self.status_bar.showMessage(f"Error saving web content: {str(e)}", 3000)

    def create_web_browsing_tab(self):
        """Create the web browsing tab"""
        web_tab = QWidget()
        layout = QVBoxLayout(web_tab)
        
        # Agent selection
        agent_layout = QHBoxLayout()
        agent_layout.addWidget(QLabel("Select Web Browsing Agent:"))
        self.web_agent_selector = QComboBox()
        
        # Filter to web browsing agents
        active_agents = self.agent_manager.get_active_agents()
        web_agents = [
            agent for agent in active_agents
            if agent["agent_type"] == "web_browsing"
        ]
        
        if web_agents:
            for agent in web_agents:
                display_name = f"{agent['agent_id']}{' (Multi-Agent)' if agent.get('multi_agent', False) else ''}"
                self.web_agent_selector.addItem(display_name, agent['agent_id'])
        else:
            self.web_agent_selector.addItem("No web browsing agents available")
        
        agent_layout.addWidget(self.web_agent_selector)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_web_agents)
        agent_layout.addWidget(refresh_button)
        
        # Create button
        create_button = QPushButton("Create New")
        create_button.clicked.connect(self.create_web_browsing_agent)
        agent_layout.addWidget(create_button)
        
        # Multi-agent toggle (if supported)
        self.multi_agent_toggle = QCheckBox("Use Multi-Agent Mode")
        self.multi_agent_toggle.setEnabled(False)
        self.multi_agent_toggle.stateChanged.connect(self.toggle_multi_agent_mode)
        agent_layout.addWidget(self.multi_agent_toggle)
        
        agent_layout.addStretch()
        layout.addLayout(agent_layout)
        
        # Create splitter for content and conversation
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top panel for web content display
        web_content_panel = QWidget()
        web_content_layout = QVBoxLayout(web_content_panel)
        
        # Create a read-only text area for showing web content
        self.web_content_display = QTextBrowser()  # Changed to QTextBrowser
        self.web_content_display.setOpenExternalLinks(True)  # Allow clicking on links
        self.web_content_display.setPlaceholderText("Web content will be displayed here")

        web_content_layout.addWidget(QLabel("Web Content:"))
        web_content_layout.addWidget(self.web_content_display)
        
        # Add save button for content
        save_button = QPushButton("Save Content")
        save_button.clicked.connect(self.save_web_content)
        web_content_layout.addWidget(save_button)
        
        # Add to splitter
        splitter.addWidget(web_content_panel)
        
        # Conversation widget
        self.web_conversation = ConversationWidget()
        splitter.addWidget(self.web_conversation)
        
        # Add splitter to layout
        layout.addWidget(splitter, stretch=1)
        
        # Input area
        input_layout = QHBoxLayout()
        self.web_input = QTextEdit()
        self.web_input.setPlaceholderText("Enter your web search or browsing task...")
        self.web_input.setMaximumHeight(100)
        input_layout.addWidget(self.web_input)
        
        self.web_send_button = QPushButton("Send")
        self.web_send_button.clicked.connect(self.send_web_command)
        if self.web_agent_selector.currentText() == "No web browsing agents available":
            self.web_send_button.setEnabled(False)
        input_layout.addWidget(self.web_send_button)
        layout.addLayout(input_layout)
        
        # Connect signals
        self.web_agent_selector.currentIndexChanged.connect(self.on_web_agent_selected)
        
        self.tabs.addTab(web_tab, "Web Browsing")
        
    def send_web_command(self):
        """Send command to web browsing agent - simplified to just perform search"""
        if self.web_agent_selector.currentText() == "No web browsing agents available":
            return
        
        # Get agent ID from combo box data
        agent_id = self.web_agent_selector.itemData(self.web_agent_selector.currentIndex())
        if not agent_id:
            return
        
        # Get command
        command = self.web_input.toPlainText().strip()
        if not command:
            return
        
        # Add to conversation
        self.web_conversation.add_message(command, is_user=True)
        
        # Clear input
        self.web_input.clear()
        
        # Disable input while processing
        self.web_input.setEnabled(False)
        self.web_send_button.setEnabled(False)
        
        # Show processing status
        self.status_bar.showMessage("Searching the web, please wait...", 0)
        
        # Clear the web content display and show loading message
        self.web_content_display.setText("Searching the web...")
        
        # Run agent in thread
        self.run_agent_in_thread(
            agent_id,
            command,
            self.handle_web_result
        )

    def handle_web_result(self, result: str):
        """Handle web browsing agent result - display markdown content with proper rendering
        
        Args:
            result: Agent result
        """
        # Add to conversation
        self.web_conversation.add_message(result, is_user=False)
        
        try:
            # Check if the result looks like it contains markdown
            contains_markdown = any(marker in result for marker in ['#', '```', '*', '[', '|', '>'])
            
            if contains_markdown:
                # Convert markdown to HTML
                html_content = self.convert_markdown_to_html(result)
                self.web_content_display.setHtml(html_content)
            else:
                # Fall back to the existing HTML conversion for simple search results
                html_content = "<html><body style='font-family: Arial, sans-serif;'>"
                lines = result.split("\n")
                
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Handle title line
                    if "Search Results for:" in line:
                        html_content += f"<h2>{line}</h2>"
                        i += 1
                        continue
                        
                    # Check if this line is a title and the next is a URL
                    if i < len(lines) - 1:
                        next_line = lines[i+1].strip()
                        # If current line is not a URL but the next line is
                        if (not line.startswith("http")) and next_line.startswith("http"):
                            # This is a title followed by URL
                            html_content += f"<p><b>{line}</b><br>"
                            html_content += f"<a href='{next_line}' style='color: #0066cc;'>{next_line}</a></p>"
                            i += 2  # Skip both lines
                            continue
                    
                    # Regular text line
                    if line:
                        # Check if this is a URL
                        if line.startswith("http"):
                            html_content += f"<p><a href='{line}' style='color: #0066cc;'>{line}</a></p>"
                        else:
                            html_content += f"<p>{line}</p>"
                    
                    i += 1
                
                html_content += "</body></html>"
                
                # Set content as HTML
                self.web_content_display.setHtml(html_content)
        
        except Exception as e:
            # Fallback if HTML processing fails
            self.web_content_display.setText(result)
            self.logger.error(f"Error formatting web results: {str(e)}")
        
        # Enable UI elements
        self.web_input.setEnabled(True)
        self.web_send_button.setEnabled(True)
        
        # Update status
        self.status_bar.showMessage("Web search completed", 3000)

    

    def create_visual_web_tab(self):
        """Create the visual web automation tab"""
        from app.ui.components.visual_web_tab import VisualWebTab
        
        # Create tab
        self.visual_web_tab = VisualWebTab(self.agent_manager, self)
        
        # Load agents
        self.visual_web_tab.refresh_agents()
        
        # Add tab
        self.tabs.addTab(self.visual_web_tab, "Visual Web")
        
        # Add demo functionality if needed
        try:
            # Import the demo class - try/except prevents errors if file doesn't exist yet
            from app.utils.visual_web_demo import VisualWebDemo
            VisualWebDemo.add_demo_menu(self.visual_web_tab)
        except ImportError:
            # If the demo module isn't available, don't add the demo menu
            self.logger.info("Visual Web Demo not available, skipping demo menu")

    def create_code_gen_tab(self):
        """Create the code generation tab"""
        from app.ui.components.code_gen_tab import CodeGenTab
        
        # Create tab
        self.code_gen_tab = CodeGenTab(self.agent_manager, self)
        
        # Load agents
        self.code_gen_tab.load_agents()
        
        # Add tab
        self.tabs.addTab(self.code_gen_tab, "Code Generation")

    def load_web_agents(self):
        """Load web browsing agents"""
        # Clear existing items
        self.web_agent_selector.clear()
        
        # Filter to web browsing agents
        active_agents = self.agent_manager.get_active_agents()
        web_agents = [
            agent for agent in active_agents
            if agent["agent_type"] == "web_browsing"
        ]
        
        if web_agents:
            for agent in web_agents:
                display_name = f"{agent['agent_id']}{' (Multi-Agent)' if agent.get('multi_agent', False) else ''}"
                self.web_agent_selector.addItem(display_name, agent['agent_id'])
            self.web_send_button.setEnabled(True)
            
            # Select first agent
            self.on_web_agent_selected(0)
        else:
            self.web_agent_selector.addItem("No web browsing agents available")
            self.web_send_button.setEnabled(False)
            self.multi_agent_toggle.setEnabled(False)
        
    def create_web_browsing_agent(self):
        """Create a new web browsing agent with enhanced UI display capabilities"""
        from app.ui.dialogs.create_agent_dialog import CreateAgentDialog
        
        # Create dialog
        dialog = CreateAgentDialog(self.agent_manager, self)
        
        # Set agent type to web_browsing
        index = dialog.agent_type_combo.findText("web_browsing")
        if index >= 0:
            dialog.agent_type_combo.setCurrentIndex(index)
        
        # Show dialog
        if dialog.exec():
            # Get agent configuration
            agent_config = dialog.get_agent_config()
            
            # Make sure we have proper configuration for web browsing
            if "DuckDuckGoSearchTool" not in agent_config["tools"]:
                agent_config["tools"].append("DuckDuckGoSearchTool")
            
            # Create agent
            agent_id = self.agent_manager.create_agent(
                agent_id=agent_config["agent_id"],
                agent_type=agent_config["agent_type"],
                model_config=agent_config["model_config"],
                tools=agent_config["tools"],
                additional_config=agent_config["additional_config"]
            )
            
            if agent_id:
                # Show success message
                multi_agent_msg = " (Multi-Agent)" if agent_config["additional_config"].get("multi_agent", False) else ""
                self.status_bar.showMessage(f"Web browsing agent '{agent_id}'{multi_agent_msg} created", 3000)
                
                # Reload agents
                self.load_web_agents()
                
                # Set as current agent
                for i in range(self.web_agent_selector.count()):
                    if self.web_agent_selector.itemData(i) == agent_id:
                        self.web_agent_selector.setCurrentIndex(i)
                        break
                
                # Switch to the web browsing tab to show the new agent
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) == "Web Browsing":
                        self.tabs.setCurrentIndex(i)
                        break

    # In app/ui/main_window.py
    def create_visual_web_agent(self, use_recommended_model=False):
        """Create a new visual web agent with enhanced functionality"""
        from app.ui.dialogs.create_agent_dialog import CreateAgentDialog
        
        # Create dialog
        dialog = CreateAgentDialog(self.agent_manager, self)
        
        # Set agent type to visual_web
        index = dialog.agent_type_combo.findText("visual_web")
        if index >= 0:
            dialog.agent_type_combo.setCurrentIndex(index)
        
        # Use recommended model if specified
        if use_recommended_model:
            recommended_model = "meta-llama/Llama-3.2-3B-Instruct"
            # Find the model in the list or add it
            found = False
            for i in range(dialog.model_list.count()):
                item = dialog.model_list.item(i)
                if item.text() == recommended_model:
                    dialog.model_list.setCurrentItem(item)
                    found = True
                    break
                    
            if not found:
                from PyQt6.QtWidgets import QListWidgetItem
                from PyQt6.QtCore import Qt
                item = QListWidgetItem(recommended_model)
                item.setData(Qt.ItemDataRole.UserRole, {"id": recommended_model, "is_cached": False})
                dialog.model_list.addItem(item)
                dialog.model_list.setCurrentItem(item)
        
        # Show Chrome requirement message
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Chrome Requirement",
            "Visual Web Automation requires Google Chrome to be installed on your system. "
            "Please make sure Chrome is installed before creating the agent.",
            QMessageBox.StandardButton.Ok
        )
        
        # Show dialog
        if dialog.exec():
            # Get agent configuration
            agent_config = dialog.get_agent_config()
            
            # Set flatten_messages_as_text to False for proper image handling
            if "additional_config" not in agent_config:
                agent_config["additional_config"] = {}
            agent_config["additional_config"]["flatten_messages_as_text"] = False
            
            # Create agent
            agent_id = self.agent_manager.create_agent(
                agent_id=agent_config["agent_id"],
                agent_type=agent_config["agent_type"],
                model_config=agent_config["model_config"],
                tools=agent_config["tools"],
                additional_config=agent_config["additional_config"]
            )
            
            if agent_id:
                # Show success message
                self.status_bar.showMessage(f"Visual web agent '{agent_id}' created", 3000)
                
                # Reload agents
                self.visual_web_tab.refresh_agents()
                
                # Switch to visual web tab
                self.tabs.setCurrentWidget(self.visual_web_tab)
            else:
                QMessageBox.warning(
                    self,
                    "Agent Creation Failed",
                    "Failed to create visual web agent. Please check the logs for details."
                )

    def create_code_generation_agent(self):
        """Create a new code generation agent"""
        from app.ui.dialogs.create_agent_dialog import CreateAgentDialog
        
        # Create dialog
        dialog = CreateAgentDialog(self.agent_manager, self)
        
        # Set agent type to code_generation
        index = dialog.agent_type_combo.findText("code_generation")
        if index >= 0:
            dialog.agent_type_combo.setCurrentIndex(index)
        
        # Show dialog
        if dialog.exec():
            # Get agent configuration
            agent_config = dialog.get_agent_config()
            
            # Create agent
            agent_id = self.agent_manager.create_agent(
                agent_id=agent_config["agent_id"],
                agent_type=agent_config["agent_type"],
                model_config=agent_config["model_config"],
                tools=agent_config["tools"],
                additional_config=agent_config["additional_config"]
            )
            
            if agent_id:
                # Show success message
                self.status_bar.showMessage(f"Code generation agent '{agent_id}' created", 3000)
                
                # Reload agents
                self.code_gen_tab.load_agents()
                
                # Switch to code generation tab
                self.tabs.setCurrentWidget(self.code_gen_tab)
    


    def create_media_gen_tab(self):
        """Create the media generation tab"""
        try:
            # Import needs to be here to avoid circular imports
            from app.ui.components.media_gen_tab import MediaGenTab
            
            # Create tab as a direct child of the main window, not the tab widget
            self.media_gen_tab = MediaGenTab(self.agent_manager, self)
            
            # Load agents
            self.media_gen_tab.load_agents()
            
            # Add tab to the tab widget
            self.tabs.addTab(self.media_gen_tab, "Media Generation")
            
            self.logger.info("Media generation tab created successfully")
        except Exception as e:
            self.logger.error(f"Error creating media generation tab: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def create_media_generation_agent(self):
        """Create a new media generation agent with enhanced capabilities"""
        from app.ui.dialogs.create_agent_dialog import CreateAgentDialog
        
        # Create dialog
        dialog = CreateAgentDialog(self.agent_manager, self)
        
        # Set agent type to media_generation
        index = dialog.agent_type_combo.findText("media_generation")
        if index >= 0:
            dialog.agent_type_combo.setCurrentIndex(index)
        
        # Show note about video generation capabilities
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Media Generation Capabilities",
            "The media generation agent can create both images and videos from text prompts.\n\n"
            "If in-case media (image or video) preview is not loaded then goto defined path to "
            "get the generated media (as shown in user interface)."
        )
        
        # Show dialog
        if dialog.exec():
            # Get agent configuration
            agent_config = dialog.get_agent_config()
            
            # Ensure we have the required configurations
            if "additional_config" not in agent_config:
                agent_config["additional_config"] = {}
            
            if "image_space_id" not in agent_config["additional_config"] or not agent_config["additional_config"]["image_space_id"]:
                agent_config["additional_config"]["image_space_id"] = "black-forest-labs/FLUX.1-schnell"
            
            if "video_space_id" not in agent_config["additional_config"] or not agent_config["additional_config"]["video_space_id"]:
                agent_config["additional_config"]["video_space_id"] = "SahaniJi/Instant-Video"
            
            # Create agent
            agent_id = self.agent_manager.create_agent(
                agent_id=agent_config["agent_id"],
                agent_type=agent_config["agent_type"],
                model_config=agent_config["model_config"],
                tools=agent_config["tools"],
                additional_config=agent_config["additional_config"]
            )
            
            if agent_id:
                # Show success message
                self.status_bar.showMessage(f"Media generation agent '{agent_id}' created", 3000)
                
                # Reload agents
                self.media_gen_tab.load_agents()
                
                # Switch to media generation tab
                self.tabs.setCurrentWidget(self.media_gen_tab)
            else:
                QMessageBox.warning(
                    self,
                    "Agent Creation Failed",
                    "Failed to create media generation agent. Check the logs for details."
                )

        

    

    def run_agent_in_thread(self, agent_id: str, input_text: str, callback: Callable[[str], None]):
        """Run an agent in a thread
        
        Args:
            agent_id: ID of the agent to run
            input_text: Input text for the agent
            callback: Callback function for the result
        """
        # Create and start thread
        self.agent_thread = AgentThread(self.agent_manager, agent_id, input_text)
        self.agent_thread.result_ready.connect(callback)
        self.agent_thread.progress_update.connect(lambda text: self.status_bar.showMessage(f"Processing: {text}"))
        self.agent_thread.start()
    

    def create_fine_tuning_tab(self):
        """Create the fine-tuning tab"""
        from app.ui.components.fine_tuning_tab import FineTuningTab
        
        # Create tab
        self.fine_tuning_tab = FineTuningTab(self.agent_manager, self)
        
        # Load agents
        self.fine_tuning_tab.load_agents()
        
        # Add tab
        self.tabs.addTab(self.fine_tuning_tab, "Fine-Tuning")

    # Add this method to the MainWindow class
    def create_fine_tuning_agent(self):
        """Create a new fine-tuning agent"""
        from app.ui.dialogs.create_agent_dialog import CreateAgentDialog
        
        # Create dialog
        dialog = CreateAgentDialog(self.agent_manager, self)
        
        # Set agent type to fine_tuning
        index = dialog.agent_type_combo.findText("fine_tuning")
        if index >= 0:
            dialog.agent_type_combo.setCurrentIndex(index)
        
        # Show dialog
        if dialog.exec():
            # Get agent configuration
            agent_config = dialog.get_agent_config()
            
            # Create agent
            agent_id = self.agent_manager.create_agent(
                agent_id=agent_config["agent_id"],
                agent_type=agent_config["agent_type"],
                model_config=agent_config["model_config"],
                tools=agent_config["tools"],
                additional_config=agent_config["additional_config"]
            )
            
            if agent_id:
                # Show success message
                self.status_bar.showMessage(f"Fine-tuning agent '{agent_id}' created", 3000)
                
                # Reload agents
                self.fine_tuning_tab.load_agents()
                
                # Switch to fine-tuning tab
                self.tabs.setCurrentWidget(self.fine_tuning_tab)


    def create_rag_tab(self):
        """Create the RAG tab for document retrieval and question answering"""
        from app.ui.components.rag_tab import RagTab
        
        # Create tab
        self.rag_tab = RagTab(self.agent_manager, self)
        
        # Load agents
        self.rag_tab.load_agents()
        
        # Add tab
        self.tabs.addTab(self.rag_tab, "RAG")

    def create_rag_agent(self):
        """Create a new RAG agent"""
        from app.ui.dialogs.create_agent_dialog import CreateAgentDialog
        
        # Create dialog
        dialog = CreateAgentDialog(self.agent_manager, self)
        
        # Set agent type to rag
        index = dialog.agent_type_combo.findText("rag")
        if index >= 0:
            dialog.agent_type_combo.setCurrentIndex(index)
        
        # Show dialog
        if dialog.exec():
            # Get agent configuration
            agent_config = dialog.get_agent_config()
            
            # Ensure we have the required configurations
            if "additional_config" not in agent_config:
                agent_config["additional_config"] = {}
            
            if "rag_space_id" not in agent_config["additional_config"]:
                agent_config["additional_config"]["rag_space_id"] = "AdrienB134/rag_ColPali_Qwen2VL"
            
            if "fallback_space_id" not in agent_config["additional_config"]:
                agent_config["additional_config"]["fallback_space_id"] = "openfree/PDF-RAG"
            
            # Create agent
            agent_id = self.agent_manager.create_agent(
                agent_id=agent_config["agent_id"],
                agent_type=agent_config["agent_type"],
                model_config=agent_config["model_config"],
                tools=agent_config["tools"],
                additional_config=agent_config["additional_config"]
            )
            
            if agent_id:
                # Show success message
                self.status_bar.showMessage(f"RAG agent '{agent_id}' created", 3000)
                
                # Reload agents
                self.rag_tab.load_agents()
                
                # Switch to RAG tab
                self.tabs.setCurrentWidget(self.rag_tab)
            else:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Agent Creation Failed",
                    "Failed to create RAG agent. Check the logs for details."
                )
# Import and apply the main window enhancements
try:
    from app.utils.main_window_enhancements import enhance_main_window
    # Apply enhancements to the MainWindow class
    MainWindow = enhance_main_window(MainWindow)
except ImportError:
    # If the enhancements aren't available, continue without them
    pass