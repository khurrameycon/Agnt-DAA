"""
Create Agent Dialog for SagaX1
Dialog for creating new agents
"""

import os
import logging
import uuid
from typing import Dict, List, Any, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QTabWidget, QWidget, QListWidget,
    QListWidgetItem, QSpinBox, QDoubleSpinBox, QDialogButtonBox,
    QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize

from app.core.agent_manager import AgentManager
from app.core.model_manager import ModelManager

class CreateAgentDialog(QDialog):
    """Dialog for creating a new agent"""
    
    def __init__(self, agent_manager: AgentManager, parent=None):
        """Initialize the create agent dialog
        
        Args:
            agent_manager: Agent manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.agent_manager = agent_manager
        self.model_manager = agent_manager.model_manager
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("Create New Agent")
        self.resize(600, 500)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Create basic tab
        self.create_basic_tab()
        
        # Create model tab
        self.create_model_tab()
        
        # Create tools tab
        self.create_tools_tab()
        
        # Create advanced tab
        self.create_advanced_tab()
        
        # Create agent-specific tab
        self.create_agent_specific_tab()
        
        # Create button box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        
        # Connect signals
        self.model_search_button.clicked.connect(self.search_models)
        self.agent_type_combo.currentTextChanged.connect(self.on_agent_type_changed)
        
        # Initialize UI
        self.load_available_models()
        self.load_available_tools()
        
    def create_basic_tab(self):
        """Create basic configuration tab"""
        basic_tab = QWidget()
        layout = QFormLayout(basic_tab)
        
        # Agent name
        self.agent_name_edit = QLineEdit()
        self.agent_name_edit.setPlaceholderText("Enter agent name")
        layout.addRow("Agent Name:", self.agent_name_edit)
        
        # Agent type
        self.agent_type_combo = QComboBox()
        self.agent_type_combo.addItems(self.agent_manager.get_available_agent_types())
        layout.addRow("Agent Type:", self.agent_type_combo)
        
        # Agent description
        self.agent_description_edit = QLineEdit()
        self.agent_description_edit.setPlaceholderText("Enter agent description")
        layout.addRow("Description:", self.agent_description_edit)
        
        self.tabs.addTab(basic_tab, "Basic")
    
    def create_model_tab(self):
        """Create model configuration tab"""
        model_tab = QWidget()
        layout = QVBoxLayout(model_tab)
        
        # Model search
        search_layout = QHBoxLayout()
        self.model_search_edit = QLineEdit()
        self.model_search_edit.setPlaceholderText("Search for models")
        search_layout.addWidget(self.model_search_edit)
        
        self.model_search_button = QPushButton("Search")
        search_layout.addWidget(self.model_search_button)
        
        layout.addLayout(search_layout)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.model_list)
        
        # Model parameters group
        params_group = QGroupBox("Model Parameters")
        params_layout = QFormLayout(params_group)
        
        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.1)
        params_layout.addRow("Temperature:", self.temperature_spin)
        
        # Max tokens
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 8000)
        self.max_tokens_spin.setSingleStep(10)
        self.max_tokens_spin.setValue(2048)
        params_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        params_layout.addRow("Device:", self.device_combo)
        
        layout.addWidget(params_group)
        self.tabs.addTab(model_tab, "Model")
    
    def create_tools_tab(self):
        """Create tools configuration tab"""
        tools_tab = QWidget()
        layout = QVBoxLayout(tools_tab)
        
        layout.addWidget(QLabel("Select Tools for Agent:"))
        
        # Tools list
        self.tools_list = QListWidget()
        self.tools_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.tools_list)
        
        self.tabs.addTab(tools_tab, "Tools")
    
    def create_advanced_tab(self):
        """Create advanced configuration tab"""
        advanced_tab = QWidget()
        layout = QFormLayout(advanced_tab)
        
        # Authorized imports (for CodeAgent)
        layout.addWidget(QLabel("Authorized Imports:"))
        self.authorized_imports_edit = QLineEdit()
        self.authorized_imports_edit.setPlaceholderText("numpy,pandas,matplotlib,etc")
        layout.addRow("", self.authorized_imports_edit)
        
        # Memory
        self.max_history_spin = QSpinBox()
        self.max_history_spin.setRange(10, 1000)
        self.max_history_spin.setSingleStep(10)
        self.max_history_spin.setValue(100)
        layout.addRow("Max History:", self.max_history_spin)
        
        # Set as default
        self.default_agent_check = QCheckBox("Set as default agent")
        layout.addRow("", self.default_agent_check)
        
        self.tabs.addTab(advanced_tab, "Advanced")
    
    def create_agent_specific_tab(self):
        """Create agent-specific configuration tab"""
        agent_specific_tab = QWidget()
        self.agent_specific_layout = QVBoxLayout(agent_specific_tab)
        
        # Web Browsing Agent Options
        self.web_browsing_options = QGroupBox("Web Browsing Agent Options")
        web_browsing_layout = QFormLayout(self.web_browsing_options)
        
        # Multi-agent option
        self.multi_agent_check = QCheckBox("Use multi-agent architecture")
        self.multi_agent_check.setToolTip("Enable to use a planner and browser agent working together")
        web_browsing_layout.addRow("", self.multi_agent_check)
        
        # Add description of multi-agent
        multi_agent_desc = QLabel(
            "Multi-agent architecture uses a planner agent to break down complex tasks\n"
            "and a browser agent to execute the plan. This is recommended for complex\n"
            "web tasks that involve multiple steps or websites."
        )
        multi_agent_desc.setWordWrap(True)
        web_browsing_layout.addRow("", multi_agent_desc)
        
        self.agent_specific_layout.addWidget(self.web_browsing_options)
        self.web_browsing_options.setVisible(False)
        
        # Visual Web Agent Options
        self.visual_web_options = QGroupBox("Visual Web Agent Options")
        visual_web_layout = QFormLayout(self.visual_web_options)
        
        # Browser dimensions
        self.browser_width_spin = QSpinBox()
        self.browser_width_spin.setRange(800, 1920)
        self.browser_width_spin.setSingleStep(50)
        self.browser_width_spin.setValue(1280)
        visual_web_layout.addRow("Browser Width:", self.browser_width_spin)
        
        self.browser_height_spin = QSpinBox()
        self.browser_height_spin.setRange(600, 1080)
        self.browser_height_spin.setSingleStep(50)
        self.browser_height_spin.setValue(800)
        visual_web_layout.addRow("Browser Height:", self.browser_height_spin)
        
        self.agent_specific_layout.addWidget(self.visual_web_options)
        self.visual_web_options.setVisible(False)
        
        # Code Generation Agent Options
        self.code_gen_options = QGroupBox("Code Generation Agent Options")
        code_gen_layout = QFormLayout(self.code_gen_options)
        
        # Sandbox option
        self.sandbox_check = QCheckBox("Run code in sandbox (safer but more limited)")
        self.sandbox_check.setChecked(True)
        code_gen_layout.addRow("", self.sandbox_check)
        
        self.agent_specific_layout.addWidget(self.code_gen_options)
        self.code_gen_options.setVisible(False)
        
        # Add to tabs
        self.tabs.addTab(agent_specific_tab, "Agent Options")


        # For app/ui/dialogs/create_agent_dialog.py
        # Add the media generation options to the create_agent_specific_tab method after the code_gen_options section:

        # Media Generation Agent Options
        self.media_gen_options = QGroupBox("Media Generation Agent Options")
        media_gen_layout = QFormLayout(self.media_gen_options)

        # Fine-tuning Agent Options
        self.fine_tuning_options = QGroupBox("Fine-tuning Agent Options")
        fine_tuning_layout = QFormLayout(self.fine_tuning_options)

        # Output directory
        self.ft_output_dir_layout = QHBoxLayout()
        self.ft_output_dir_edit = QLineEdit("./fine_tuned_models")
        self.ft_output_dir_layout.addWidget(self.ft_output_dir_edit)

        self.ft_output_dir_button = QPushButton("Browse...")
        self.ft_output_dir_button.clicked.connect(self.browse_ft_output_dir)
        self.ft_output_dir_layout.addWidget(self.ft_output_dir_button)

        fine_tuning_layout.addRow("Output Directory:", self.ft_output_dir_layout)

        # Push to Hub checkbox
        self.ft_push_to_hub_check = QCheckBox()
        fine_tuning_layout.addRow("Push to Hub:", self.ft_push_to_hub_check)

        self.agent_specific_layout.addWidget(self.fine_tuning_options)
        self.fine_tuning_options.setVisible(False)

        # Image Space
        self.image_space_edit = QLineEdit()
        self.image_space_edit.setText("stabilityai/stable-diffusion-xl-base-1.0")
        media_gen_layout.addRow("Image Space ID:", self.image_space_edit)

        # Video Space
        self.video_space_edit = QLineEdit()
        self.video_space_edit.setText("damo-vilab/text-to-video-ms")
        media_gen_layout.addRow("Video Space ID:", self.video_space_edit)

        self.agent_specific_layout.addWidget(self.media_gen_options)
        self.media_gen_options.setVisible(False)

        
    
    def load_available_models(self):
        """Load available models"""
        # Load cached models first
        cached_models = self.model_manager.get_cached_models()
        
        for model_id in cached_models:
            item = QListWidgetItem(model_id)
            item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": True})
            self.model_list.addItem(item)
        
        # Add some popular models
        popular_models = [
            "meta-llama/Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "Gryphe/MythoMax-L2-13B"
        ]
        
        for model_id in popular_models:
            if model_id not in cached_models:
                item = QListWidgetItem(model_id)
                item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": False})
                self.model_list.addItem(item)
    
    def search_models(self):
        """Search for models"""
        query = self.model_search_edit.text().strip()
        
        if not query:
            return
        
        # Clear the list
        self.model_list.clear()
        
        # Search for models
        models = self.model_manager.search_models(query)
        
        # Add to list
        for model in models:
            item = QListWidgetItem(model["id"])
            item.setData(Qt.ItemDataRole.UserRole, model)
            self.model_list.addItem(item)
    
    def load_available_tools(self):
        """Load available tools"""
        available_tools = self.agent_manager.get_available_tools()
        
        for tool in available_tools:
            item = QListWidgetItem(f"{tool['name']}: {tool['description']}")
            item.setData(Qt.ItemDataRole.UserRole, tool)
            self.tools_list.addItem(item)
    

    def browse_ft_output_dir(self):
        """Browse for fine-tuning output directory"""
        from PyQt6.QtWidgets import QFileDialog
        
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.ft_output_dir_edit.text()
        )
        
        if dir_path:
            self.ft_output_dir_edit.setText(dir_path)

    def on_agent_type_changed(self, agent_type):
        """Handle agent type change
        
        Args:
            agent_type: New agent type
        """
        # Hide all agent-specific option groups
        self.web_browsing_options.setVisible(False)
        self.visual_web_options.setVisible(False)
        self.code_gen_options.setVisible(False)
        self.media_gen_options.setVisible(False)
        self.fine_tuning_options.setVisible(False)
        
        # Show specific options based on agent type
        if agent_type == "web_browsing":
            self.web_browsing_options.setVisible(True)
            self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "visual_web":
            self.visual_web_options.setVisible(True)
            self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "code_generation":
            self.code_gen_options.setVisible(True)
            self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "local_model":
            self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "media_generation":
            self.media_gen_options.setVisible(True)
            self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "fine_tuning":
            self.fine_tuning_options.setVisible(True)
            self.authorized_imports_edit.setEnabled(True)
        else:
            self.authorized_imports_edit.setEnabled(False)
        
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get the agent configuration from the dialog
        
        Returns:
            Agent configuration dictionary
        """
        # Get model
        model_id = None
        selected_items = self.model_list.selectedItems()
        if selected_items:
            model_data = selected_items[0].data(Qt.ItemDataRole.UserRole)
            model_id = model_data["id"]
        
        # Get tools
        selected_tools = []
        for index in range(self.tools_list.count()):
            item = self.tools_list.item(index)
            if item.isSelected():
                tool_data = item.data(Qt.ItemDataRole.UserRole)
                selected_tools.append(tool_data["name"])
        
        # Get authorized imports
        authorized_imports = []
        if self.authorized_imports_edit.text().strip():
            authorized_imports = [
                imp.strip() for imp in self.authorized_imports_edit.text().split(",")
            ]
        
        # Generate a default agent_id if none was provided
        agent_name = self.agent_name_edit.text().strip()
        agent_id = agent_name if agent_name else f"agent_{uuid.uuid4().hex[:8]}"
        
        # Base configuration
        config = {
            "agent_id": agent_id,
            "agent_type": self.agent_type_combo.currentText(),
            "model_config": {
                "model_id": model_id,
                "temperature": self.temperature_spin.value(),
                "max_tokens": self.max_tokens_spin.value(),
                "device": self.device_combo.currentText()
            },
            "tools": selected_tools,
            "additional_config": {
                "description": self.agent_description_edit.text().strip(),
                "max_history": self.max_history_spin.value(),
                "authorized_imports": authorized_imports,
                "is_default": self.default_agent_check.isChecked()
            }
        }
        
        # Add agent-specific configuration
        agent_type = self.agent_type_combo.currentText()
        
        if agent_type == "web_browsing":
            config["additional_config"]["multi_agent"] = self.multi_agent_check.isChecked()
        elif agent_type == "visual_web":
            config["additional_config"]["browser_width"] = self.browser_width_spin.value()
            config["additional_config"]["browser_height"] = self.browser_height_spin.value()
        elif agent_type == "code_generation":
            config["additional_config"]["sandbox"] = self.sandbox_check.isChecked()
        
        elif agent_type == "media_generation":
            config["additional_config"]["image_space_id"] = self.image_space_edit.text().strip()
            config["additional_config"]["video_space_id"] = self.video_space_edit.text().strip()
        elif agent_type == "fine_tuning":
            config["additional_config"]["output_dir"] = self.ft_output_dir_edit.text()
            config["additional_config"]["push_to_hub"] = self.ft_push_to_hub_check.isChecked()
        return config