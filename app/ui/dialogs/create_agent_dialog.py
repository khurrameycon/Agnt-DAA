"""
Create Agent Dialog for sagax1
Updated with agent-specific API provider logic and Anthropic preference
"""

import os
import logging
import uuid
import requests
from typing import Dict, List, Any, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QTabWidget, QWidget, QListWidget,
    QListWidgetItem, QSpinBox, QDoubleSpinBox, QDialogButtonBox,
    QGroupBox, QScrollArea, QRadioButton, QMessageBox  
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QCursor
from app.ui.dialogs.execution_mode_guide import ExecutionModeGuideDialog
from app.core.agent_manager import AgentManager
from app.core.model_manager import ModelManager

class CreateAgentDialog(QDialog):
    """Dialog for creating a new agent with agent-specific provider filtering"""
    
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
        
        self.setStyleSheet("""
        QDialog {
            color: black;
            background-color: #F9FBFD;
        }
        
        QRadioButton, QCheckBox, QLabel, QGroupBox, QComboBox {
            color: black;
        }
        
        QComboBox QAbstractItemView {
            color: black;
            background-color: white;
        }
        
        QListWidget, QListWidget::item {
            color: black;
        }
    """)

        # Track if we've loaded inference models
        self.inference_models_loaded = False
        self.inference_models = []
        
        # Define agent-specific provider mappings
        self.agent_provider_mapping = {
            "local_model": ["Local Execution", "HF API (Remote)", "OpenAI API", "Gemini API", "Groq API", "Anthropic API"],
            "web_browsing": ["OpenAI API", "Gemini API", "Groq API"], 
            "code_generation": ["Anthropic API", "Groq API", "OpenAI API", "Gemini API"],  # Anthropic first
            "rag": ["Groq API", "OpenAI API", "Gemini API"],
            "fine_tuning": ["Local Execution", "HF API (Remote)"],  # Keep existing
            "visual_web": ["Local Execution", "HF API (Remote)", "OpenAI API", "Gemini API", "Groq API", "Anthropic API"],  # No changes
            "media_generation": ["Local Execution", "HF API (Remote)", "OpenAI API", "Gemini API", "Groq API", "Anthropic API"]  # No changes
        }
        
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
        layout.setSpacing(8)
        
        # Add execution mode selection at the top
        execution_layout = QHBoxLayout()
        execution_layout.setSpacing(10)
        execution_layout.addWidget(QLabel("Select API Provider:"))
        
        self.provider_combo = QComboBox()
        # Initially populate with all providers, will be filtered by agent type
        self.provider_combo.addItems([
            "Local Execution",
            "HF API (Remote)", 
            "OpenAI API",
            "Gemini API", 
            "Groq API",
            "Anthropic API"
        ])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        execution_layout.addWidget(self.provider_combo)                            
        
        layout.addLayout(execution_layout)
        
        # Add API key info
        self.api_info = QLabel("Note: API mode is FAST but requires proper API key configuration")
        self.api_info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.api_info)
        
        # Model search - more compact layout
        search_layout = QHBoxLayout()
        search_layout.setSpacing(5)
        self.model_search_edit = QLineEdit()
        self.model_search_edit.setPlaceholderText("Search for models")
        search_layout.addWidget(self.model_search_edit, 4)
        
        self.model_search_button = QPushButton("Search")
        search_layout.addWidget(self.model_search_button, 1)
        
        layout.addLayout(search_layout)
        
        # Model list - make it larger
        model_list_label = QLabel("Select Model:")
        model_list_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(model_list_label)
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.model_list.setMinimumHeight(200)
        layout.addWidget(self.model_list, 3)
        
        # Style the model list
        self.model_list.setStyleSheet("""
            QListWidget {
                color: black;
                background-color: white;
                border: 1px solid #E0E0E0;
            }
            QListWidget::item {
                color: black;
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #4F8BC9;
                color: black;
            }
            QListWidget::item:hover {
                background-color: #E0E0E0;
                color: black;
            }
            QListWidget::item:selected:active {
                background-color: #4F8BC9;
                color: black;
            }
            QListWidget::item:selected:!active {
                background-color: #E0E0E0;
                color: black;
            }
            QListWidget::item:alternate {
                background-color: #F9F9F9;
            }
        """)
        
        # Model parameters group
        params_group = QGroupBox("Model Parameters")
        params_layout = QFormLayout(params_group)
        params_layout.setVerticalSpacing(8)
        params_layout.setHorizontalSpacing(15)
        
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
        
        # Device - only meaningful for local execution
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        params_layout.addRow("Device:", self.device_combo)
        
        layout.addWidget(params_group)
        
        self.tabs.addTab(model_tab, "Model")
    
    def on_agent_type_changed(self, agent_type):
        """Handle agent type change - filter providers based on agent type
        
        Args:
            agent_type: New agent type
        """
        # Get allowed providers for this agent type
        allowed_providers = self.agent_provider_mapping.get(agent_type, [
            "Local Execution", "HF API (Remote)", "OpenAI API", "Gemini API", "Groq API", "Anthropic API"
        ])
        
        # Clear and repopulate provider combo
        self.provider_combo.clear()
        
        # For code generation agents, check if Anthropic key is available
        if agent_type == "code_generation":
            anthropic_key = self.agent_manager.config_manager.get_anthropic_api_key()
            if anthropic_key:
                # If Anthropic key is available, show Anthropic first (default) but allow others
                self.provider_combo.addItems(allowed_providers)
                self.provider_combo.setCurrentText("Anthropic API")
                self.api_info.setText("Note: Anthropic API is preferred for code generation. Other APIs available as alternatives.")
                self.api_info.setStyleSheet("color: #27AE60; font-style: italic; font-weight: bold;")
            else:
                # If no Anthropic key, show alternatives and highlight the need for Anthropic
                alternative_providers = [p for p in allowed_providers if p != "Anthropic API"]
                if alternative_providers:
                    self.provider_combo.addItems(alternative_providers)
                self.api_info.setText("⚠️ Anthropic API key not found. Please add it in Settings for optimal code generation. Using alternative providers.")
                self.api_info.setStyleSheet("color: #FF9800; font-style: italic; font-weight: bold;")
        else:
            # For other agent types, just show allowed providers
            self.provider_combo.addItems(allowed_providers)
            
            # Update info message based on agent type
            if agent_type == "web_browsing":
                self.api_info.setText("Note: Web browsing agents use API providers for better performance and reliability.")
            elif agent_type == "rag":
                self.api_info.setText("Note: RAG agents use API providers for document question answering.")
            elif agent_type == "local_model":
                self.api_info.setText("Note: Local model agents support both local execution and API providers.")
            else:
                self.api_info.setText("Note: API mode is FAST but requires proper API key configuration")
            
            self.api_info.setStyleSheet("color: #666; font-style: italic;")
        
        # Set default provider based on agent type
        if allowed_providers:
            default_provider = allowed_providers[0]
            self.provider_combo.setCurrentText(default_provider)
        
        # Trigger provider change to update UI
        self.on_provider_changed(self.provider_combo.currentText())
        
        # Handle agent-specific options visibility (existing logic)
        self._handle_agent_specific_options(agent_type)
    
    def _handle_agent_specific_options(self, agent_type):
        """Handle agent-specific option visibility"""
        # Hide all agent-specific option groups
        if hasattr(self, 'web_browsing_options'):
            self.web_browsing_options.setVisible(False)
        if hasattr(self, 'visual_web_options'):
            self.visual_web_options.setVisible(False)
        if hasattr(self, 'code_gen_options'):
            self.code_gen_options.setVisible(False)
        if hasattr(self, 'media_gen_options'):
            self.media_gen_options.setVisible(False)
        if hasattr(self, 'fine_tuning_options'):
            self.fine_tuning_options.setVisible(False)
        
        # Show specific options based on agent type
        if agent_type == "web_browsing" and hasattr(self, 'web_browsing_options'):
            self.web_browsing_options.setVisible(True)
            if hasattr(self, 'authorized_imports_edit'):
                self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "visual_web" and hasattr(self, 'visual_web_options'):
            self.visual_web_options.setVisible(True)
            if hasattr(self, 'authorized_imports_edit'):
                self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "code_generation" and hasattr(self, 'code_gen_options'):
            self.code_gen_options.setVisible(True)
            if hasattr(self, 'authorized_imports_edit'):
                self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "local_model":
            if hasattr(self, 'authorized_imports_edit'):
                self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "media_generation" and hasattr(self, 'media_gen_options'):
            self.media_gen_options.setVisible(True)
            if hasattr(self, 'authorized_imports_edit'):
                self.authorized_imports_edit.setEnabled(True)
        elif agent_type == "fine_tuning" and hasattr(self, 'fine_tuning_options'):
            self.fine_tuning_options.setVisible(True)
            if hasattr(self, 'authorized_imports_edit'):
                self.authorized_imports_edit.setEnabled(True)
        else:
            if hasattr(self, 'authorized_imports_edit'):
                self.authorized_imports_edit.setEnabled(False)

    def on_provider_changed(self, provider):
        """Handle provider change"""
        is_local = provider == "Local Execution"
        is_hf_api = provider == "HF API (Remote)"
        
        # Enable/disable controls based on selection
        self.device_combo.setEnabled(is_local)
        
        # Update model search behavior
        current_agent_type = self.agent_type_combo.currentText()
        should_disable_search = (
            is_hf_api and current_agent_type in ["web_browsing", "code_generation", "rag"]
        )
        
        self.model_search_edit.setEnabled(not should_disable_search)
        self.model_search_button.setEnabled(not should_disable_search)
        
        if should_disable_search:
            self.model_search_edit.setPlaceholderText(f"Model search disabled for {current_agent_type} agents")
            self.model_search_edit.clear()
        elif not is_local and not is_hf_api:
            self.model_search_edit.setPlaceholderText("Search disabled for API providers - predefined models available")
            self.model_search_edit.setEnabled(False)
            self.model_search_button.setEnabled(False)
            # Load predefined models for API providers
            self._load_provider_models(provider)
        else:
            self.model_search_edit.setPlaceholderText("Search for models")
            if is_local:
                self.load_available_models()  # Load local models
            elif is_hf_api:
                self.load_available_models()  # Load HF API models if allowed

    def _load_provider_models(self, provider):
        """Load default models for API providers"""
        self.model_list.clear()
        
        provider_models = {
            "OpenAI API": ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "Gemini API": ["gemini-2.5-pro-preview-05-06", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
            "Groq API": ["deepseek-r1-distill-llama-70b", "qwen-qwq-32b", "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
            "Anthropic API": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        }
        
        models = provider_models.get(provider, [])
        for model_id in models:
            item = QListWidgetItem(model_id)
            item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": False})
            self.model_list.addItem(item)

    def load_available_models(self):
        """Load available models based on the selected provider and agent type"""
        # Clear the list
        self.model_list.clear()
        
        # Get current provider and agent type
        provider_text = self.provider_combo.currentText()
        agent_type = self.agent_type_combo.currentText()
        
        # Check if this agent type should not use HF API
        if provider_text == "HF API (Remote)" and agent_type in ["web_browsing", "code_generation", "rag"]:
            # Show message that HF is not available for this agent type
            no_hf_item = QListWidgetItem(f"HF API not available for {agent_type} agents")
            no_hf_item.setFlags(no_hf_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_list.addItem(no_hf_item)
            return
        
        # Show different models based on provider selection
        if provider_text in ["OpenAI API", "Gemini API", "Groq API", "Anthropic API"]:
            # Load API provider models
            self._load_provider_models(provider_text)
        elif provider_text == "HF API (Remote)":
            # Show "Loading..." while fetching inference models (only for allowed agent types)
            loading_item = QListWidgetItem("Loading inference models...")
            loading_item.setFlags(loading_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_list.addItem(loading_item)
            
            # Load inference models if we haven't already
            if not self.inference_models_loaded:
                QTimer.singleShot(100, self.fetch_inference_models)
            else:
                # If we already loaded them, just populate the list
                self.populate_inference_models()
        else:
            # Local execution mode - load cached and popular models
            self._load_local_models()
    
    def search_models(self):
        """Search for models based on execution mode"""
        query = self.model_search_edit.text().strip()
        
        if not query:
            return
        
        # Clear the list
        self.model_list.clear()
        
        # Show loading item
        loading_item = QListWidgetItem(f"Searching for '{query}'...")
        loading_item.setFlags(loading_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        self.model_list.addItem(loading_item)
        
        provider_text = self.provider_combo.currentText()
        if provider_text == "HF API (Remote)":
            # For API mode, search within the loaded inference models
            QTimer.singleShot(100, lambda: self._search_inference_models(query))
        else:
            # For local mode, use the model manager search
            QTimer.singleShot(100, lambda: self._search_local_models(query))
    
    def fetch_inference_models(self):
        """Fetch available models from the Hugging Face API that support inference"""
        self.setCursor(QCursor(Qt.CursorShape.WaitCursor))
        
        try:
            # For now, use fallback models since HF API can be unreliable
            self.model_list.clear()
            
            fallback_inference_models = [
                "Qwen/QwQ-32B",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.3-70B-Instruct",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "microsoft/Phi-3-mini-4k-instruct",
                "HuggingFaceH4/zephyr-7b-beta",
                "distilbert/distilgpt2",
                "Qwen/Qwen2.5-72B-Instruct",
                "NousResearch/Hermes-3-Llama-3.1-8B",
                "CohereLabs/c4ai-command-r-plus-08-2024",
                "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
                "microsoft/DialoGPT-medium",
                "EleutherAI/gpt-neo-1.3B",
                "mistralai/Mistral-Nemo-Instruct-2407",
                "microsoft/Phi-3.5-mini-instruct"
            ]
            
            for model_id in fallback_inference_models:
                item = QListWidgetItem(model_id)
                item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": False})
                self.model_list.addItem(item)
                
            self.inference_models_loaded = True
                
        except Exception as e:
            self.model_list.clear()
            error_item = QListWidgetItem(f"Error fetching models: {str(e)}")
            error_item.setFlags(error_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_list.addItem(error_item)
            
            self.logger.error(f"Error fetching inference models: {str(e)}")
        
        finally:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
    def populate_inference_models(self):
        """Populate the model list with available inference models"""
        self.model_list.clear()
        
        if not self.inference_models:
            no_models_item = QListWidgetItem("No inference models found")
            no_models_item.setFlags(no_models_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_list.addItem(no_models_item)
            return
        
        # Add each model to the list
        for model in self.inference_models:
            model_id = model["id"]
            item = QListWidgetItem(model_id)
            item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": False})
            
            # Add more details in tooltip if available
            tooltip = f"Model: {model_id}"
            if "downloads" in model:
                tooltip += f"\nDownloads: {model['downloads']:,}"
            
            item.setToolTip(tooltip)
            self.model_list.addItem(item)
    
    def _load_local_models(self):
        """Load cached models for local execution"""
        # Load cached models first
        cached_models = self.model_manager.get_cached_models()
        
        for model_id in cached_models:
            item = QListWidgetItem(model_id)
            item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": True})
            # Add (cached) tag to the text
            item.setText(f"{model_id} (cached)")
            self.model_list.addItem(item)
        
        # Add some popular chat models
        popular_models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "Qwen/Qwen1.5-0.5B-Chat",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ]
        
        for model_id in popular_models:
            if model_id not in cached_models:
                item = QListWidgetItem(model_id)
                item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": False})
                self.model_list.addItem(item)
    
    def _search_inference_models(self, query):
        """Search for models with inference API support
        
        Args:
            query: Search query
        """
        self.setCursor(QCursor(Qt.CursorShape.WaitCursor))
        
        try:
            # Directly search Hugging Face API for inference models
            url = f"https://huggingface.co/api/models?search={query}&inference_endpoints=true"
            response = requests.get(url)
            
            self.model_list.clear()
            
            if response.status_code == 200:
                models = response.json()
                
                if not models:
                    no_results = QListWidgetItem(f"No inference models found for '{query}'")
                    no_results.setFlags(no_results.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                    self.model_list.addItem(no_results)
                    return
                
                # Add each model to the list
                for model in models:
                    model_id = model["id"]
                    item = QListWidgetItem(model_id)
                    item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": False})
                    
                    # Add more details in tooltip if available
                    tooltip = f"Model: {model_id}"
                    if "downloads" in model:
                        tooltip += f"\nDownloads: {model['downloads']:,}"
                    
                    item.setToolTip(tooltip)
                    self.model_list.addItem(item)
            else:
                error_item = QListWidgetItem(f"Error searching models: {response.status_code}")
                error_item.setFlags(error_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self.model_list.addItem(error_item)
        
        except Exception as e:
            self.model_list.clear()
            error_item = QListWidgetItem(f"Error searching models: {str(e)}")
            error_item.setFlags(error_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_list.addItem(error_item)
            self.logger.error(f"Error searching inference models: {str(e)}")
        
        finally:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
    def _search_local_models(self, query):
        """Search for models using the model manager
        
        Args:
            query: Search query
        """
        try:
            # Search for models using the model manager
            models = self.model_manager.search_models(query)
            
            # Clear the list
            self.model_list.clear()
            
            if not models:
                no_results = QListWidgetItem(f"No models found for '{query}'")
                no_results.setFlags(no_results.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self.model_list.addItem(no_results)
                return
            
            # Get cached models for reference
            cached_models = self.model_manager.get_cached_models()
            
            # Add to list
            for model in models:
                model_id = model["id"]
                is_cached = model_id in cached_models
                
                item = QListWidgetItem(model_id + (" (cached)" if is_cached else ""))
                item.setData(Qt.ItemDataRole.UserRole, {"id": model_id, "is_cached": is_cached})
                self.model_list.addItem(item)
                
        except Exception as e:
            self.model_list.clear()
            error_item = QListWidgetItem(f"Error searching models: {str(e)}")
            error_item.setFlags(error_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_list.addItem(error_item)
            self.logger.error(f"Error searching local models: {str(e)}")
    
    def load_available_tools(self):
        """Load available tools"""
        available_tools = self.agent_manager.get_available_tools()
        
        for tool in available_tools:
            item = QListWidgetItem(f"{tool['name']}: {tool['description']}")
            item.setData(Qt.ItemDataRole.UserRole, tool)
            self.tools_list.addItem(item)
    
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
        
        # Code Generation Space
        self.code_space_edit = QLineEdit("sitammeur/Qwen-Coder-llamacpp")
        code_gen_layout.addRow("Code Space ID:", self.code_space_edit)
        
        self.agent_specific_layout.addWidget(self.code_gen_options)
        self.code_gen_options.setVisible(False)
        
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
        
        # Add to tabs
        self.tabs.addTab(agent_specific_tab, "Agent Options")
    
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
        
        # Determine provider and execution mode
        provider_text = self.provider_combo.currentText()
        provider_map = {
            "Local Execution": ("local", True, False),
            "HF API (Remote)": ("huggingface", False, True),
            "OpenAI API": ("openai", False, True),
            "Gemini API": ("gemini", False, True),
            "Groq API": ("groq", False, True),
            "Anthropic API": ("anthropic", False, True)  # Add Anthropic mapping
        }
        
        api_provider, use_local, use_api = provider_map.get(provider_text, ("local", True, False))
        
        # Base configuration
        config = {
            "agent_id": agent_id,
            "agent_type": self.agent_type_combo.currentText(),
            "model_config": {
                "model_id": model_id,
                "temperature": self.temperature_spin.value(),
                "max_tokens": self.max_tokens_spin.value(),
                "device": self.device_combo.currentText(),
                "use_local_execution": use_local,
                "use_api": use_api,
                "api_provider": api_provider
            },
            "tools": selected_tools,
            "additional_config": {
                "description": self.agent_description_edit.text().strip(),
                "max_history": self.max_history_spin.value(),
                "authorized_imports": authorized_imports,
                "is_default": self.default_agent_check.isChecked(),
                "use_local_execution": use_local,
                "use_api": use_api,
                "api_provider": api_provider
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
            config["additional_config"]["code_space_id"] = self.code_space_edit.text().strip()
        elif agent_type == "media_generation":
            config["additional_config"]["image_space_id"] = self.image_space_edit.text().strip()
            config["additional_config"]["video_space_id"] = self.video_space_edit.text().strip()
        elif agent_type == "fine_tuning":
            config["additional_config"]["output_dir"] = self.ft_output_dir_edit.text()
            config["additional_config"]["push_to_hub"] = self.ft_push_to_hub_check.isChecked()
        
        return config