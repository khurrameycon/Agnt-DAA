"""
Fine-tuning Tab Component for SagaX1
Tab for fine-tuning models on custom datasets
"""

import os
import json
import logging
from typing import Optional, Dict, List, Any
import tempfile

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QTabWidget,
    QCheckBox, QFileDialog, QLineEdit, QSpinBox, 
    QDoubleSpinBox, QGroupBox, QFormLayout, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QFont

from app.core.agent_manager import AgentManager
from app.ui.components.conversation import ConversationWidget


class DatasetTableWidget(QWidget):
    """Widget for creating and editing instruction datasets"""
    
    def __init__(self, parent=None):
        """Initialize the dataset table widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Instruction", "Input (Optional)", "Output"])
        
        # Set column stretch
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.table)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Add row button
        self.add_button = QPushButton("Add Row")
        self.add_button.clicked.connect(self.add_row)
        button_layout.addWidget(self.add_button)
        
        # Delete row button
        self.delete_button = QPushButton("Delete Row")
        self.delete_button.clicked.connect(self.delete_row)
        button_layout.addWidget(self.delete_button)
        
        # Import from CSV button
        self.import_button = QPushButton("Import from CSV")
        self.import_button.clicked.connect(self.import_from_csv)
        button_layout.addWidget(self.import_button)
        
        # Export to CSV button
        self.export_button = QPushButton("Export to CSV")
        self.export_button.clicked.connect(self.export_to_csv)
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
    
    def add_row(self):
        """Add a new row to the table"""
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
    
    def delete_row(self):
        """Delete the selected row from the table"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            return
        
        # Get unique rows
        selected_rows = list(set([item.row() for item in selected_items]))
        
        # Remove rows in reverse order to avoid indexing issues
        for row in sorted(selected_rows, reverse=True):
            self.table.removeRow(row)
    
    def import_from_csv(self):
        """Import data from a CSV file"""
        try:
            import pandas as pd
            
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Dataset from CSV",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if not file_path:
                return
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ["instruction", "output"]
            if not all(col in df.columns for col in required_columns):
                QMessageBox.warning(
                    self,
                    "Import Error",
                    f"CSV file must contain the following columns: {', '.join(required_columns)}"
                )
                return
            
            # Clear table
            self.table.setRowCount(0)
            
            # Fill table
            for _, row in df.iterrows():
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)
                
                # Set instruction
                instruction_item = QTableWidgetItem(row["instruction"])
                self.table.setItem(row_position, 0, instruction_item)
                
                # Set input (if available)
                if "input" in df.columns:
                    input_text = row["input"] if not pd.isna(row["input"]) else ""
                    input_item = QTableWidgetItem(input_text)
                    self.table.setItem(row_position, 1, input_item)
                else:
                    self.table.setItem(row_position, 1, QTableWidgetItem(""))
                
                # Set output
                output_item = QTableWidgetItem(row["output"])
                self.table.setItem(row_position, 2, output_item)
            
            self.logger.info(f"Imported {self.table.rowCount()} rows from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error importing from CSV: {str(e)}")
            QMessageBox.warning(
                self,
                "Import Error",
                f"Error importing from CSV: {str(e)}"
            )
    
    def export_to_csv(self):
        """Export data to a CSV file"""
        try:
            import pandas as pd
            
            # Check if table is empty
            if self.table.rowCount() == 0:
                QMessageBox.warning(
                    self,
                    "Export Error",
                    "The table is empty. Nothing to export."
                )
                return
            
            # Open file dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Dataset to CSV",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if not file_path:
                return
            
            # Create data list
            data = []
            for row in range(self.table.rowCount()):
                instruction = self.table.item(row, 0).text() if self.table.item(row, 0) else ""
                input_text = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
                output = self.table.item(row, 2).text() if self.table.item(row, 2) else ""
                
                data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output
                })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Exported {len(data)} rows to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            QMessageBox.warning(
                self,
                "Export Error",
                f"Error exporting to CSV: {str(e)}"
            )
    
    def get_dataset(self) -> List[Dict[str, str]]:
        """Get the dataset as a list of dictionaries
        
        Returns:
            List of dictionaries with instruction, input, and output fields
        """
        dataset = []
        for row in range(self.table.rowCount()):
            instruction = self.table.item(row, 0).text() if self.table.item(row, 0) else ""
            input_text = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
            output = self.table.item(row, 2).text() if self.table.item(row, 2) else ""
            
            # Skip empty rows
            if not instruction or not output:
                continue
            
            dataset.append({
                "instruction": instruction,
                "input": input_text,
                "output": output
            })
        
        return dataset


class FineTuningTab(QWidget):
    """Tab for fine-tuning models"""
    
    def __init__(self, agent_manager: AgentManager, parent=None):
        """Initialize the fine-tuning tab
        
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
        
        # Create bottom panel with progress and logs
        self.create_bottom_panel()
    
    def create_top_bar(self):
        """Create top bar with agent selection and controls"""
        top_layout = QHBoxLayout()
        
        # Agent selection
        top_layout.addWidget(QLabel("Select Fine-Tuning Agent:"))
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
        """Create main panel with dataset editor and model configuration"""
        # Create tabs for dataset and configuration
        self.tabs = QTabWidget()
        
        # Dataset tab
        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(dataset_tab)
        
        # Create dataset table
        self.dataset_table = DatasetTableWidget()
        dataset_layout.addWidget(self.dataset_table)
        
        # Add example row button
        example_button = QPushButton("Add Example Row")
        example_button.clicked.connect(self.add_example_row)
        dataset_layout.addWidget(example_button)
        
        # Add dataset tab
        self.tabs.addTab(dataset_tab, "Dataset")
        
        # Configuration tab
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Create form layout for configuration
        form_layout = QFormLayout()
        
        # Model selection
        self.model_selector = QComboBox()
        self.model_selector.clear()
        self.model_selector.addItems([
            "bigscience/bloomz-1b7",  # Most reliable model for fine-tuning
            "bigscience/bloomz-560m",  # Even smaller model that's very reliable
            "facebook/opt-350m",      # Another reliable small model
            "EleutherAI/pythia-410m"  # Another reliable option
        ])

        # Then update the model_selector description with a warning
        self.model_selector.setToolTip("These models are specifically selected for reliable fine-tuning. Other models may cause errors.")

        # And add a warning label under the model selector:
        model_warning = QLabel("⚠️ Please use these pre-selected models for reliable fine-tuning. Other models may require custom configuration.")
        model_warning.setStyleSheet("color: orange;")
        form_layout.addRow("", model_warning)

        self.model_selector.setEditable(True)
        form_layout.addRow("Base Model:", self.model_selector)
        
        # Training epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 50)
        self.epochs_spin.setValue(3)
        form_layout.addRow("Training Epochs:", self.epochs_spin)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        form_layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setSingleStep(1e-6)
        self.lr_spin.setValue(2e-5)
        self.lr_spin.setDecimals(6)
        form_layout.addRow("Learning Rate:", self.lr_spin)
        
        # LoRA rank
        self.lora_r_spin = QSpinBox()
        self.lora_r_spin.setRange(1, 256)
        self.lora_r_spin.setValue(16)
        form_layout.addRow("LoRA Rank:", self.lora_r_spin)
        
        # LoRA alpha
        self.lora_alpha_spin = QSpinBox()
        self.lora_alpha_spin.setRange(1, 256)
        self.lora_alpha_spin.setValue(32)
        form_layout.addRow("LoRA Alpha:", self.lora_alpha_spin)
        
        # Output dir
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("./fine_tuned_models")
        self.output_dir_layout.addWidget(self.output_dir_edit)
        
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        self.output_dir_layout.addWidget(self.output_dir_button)
        
        form_layout.addRow("Output Directory:", self.output_dir_layout)
        
        # Hugging Face Hub Integration
        self.hf_group = QGroupBox("Hugging Face Hub Integration")
        hf_layout = QFormLayout(self.hf_group)
        
        self.push_to_hub_check = QCheckBox()
        hf_layout.addRow("Push to Hub:", self.push_to_hub_check)
        
        self.hub_model_id_edit = QLineEdit()
        self.hub_model_id_edit.setPlaceholderText("username/model-name")
        hf_layout.addRow("Hub Model ID:", self.hub_model_id_edit)
        
        # Add to layout
        config_layout.addLayout(form_layout)
        config_layout.addWidget(self.hf_group)
        config_layout.addStretch()
        
        # Add training button
        self.train_button = QPushButton("Start Fine-Tuning")
        self.train_button.clicked.connect(self.start_fine_tuning)
        config_layout.addWidget(self.train_button)
        
        # Add configuration tab
        self.tabs.addTab(config_tab, "Configuration")
        
        # Testing tab
        testing_tab = QWidget()
        testing_layout = QVBoxLayout(testing_tab)
        
        testing_form = QFormLayout()
        
        # Instruction input
        self.test_instruction_edit = QTextEdit()
        self.test_instruction_edit.setPlaceholderText("Enter instruction to test the fine-tuned model...")
        testing_form.addRow("Instruction:", self.test_instruction_edit)
        
        # Input (optional)
        self.test_input_edit = QTextEdit()
        self.test_input_edit.setPlaceholderText("Enter optional input...")
        testing_form.addRow("Input (Optional):", self.test_input_edit)
        
        # Generate button
        self.generate_button = QPushButton("Generate Response")
        self.generate_button.clicked.connect(self.generate_response)
        
        # Output display
        self.test_output_display = QTextEdit()
        self.test_output_display.setReadOnly(True)
        self.test_output_display.setPlaceholderText("Generated response will appear here...")
        testing_form.addRow("Response:", self.test_output_display)
        
        # Add to layout
        testing_layout.addLayout(testing_form)
        testing_layout.addWidget(self.generate_button)
        
        # Add testing tab
        self.tabs.addTab(testing_tab, "Test Model")
        
        # Add tabs to layout
        self.layout.addWidget(self.tabs, stretch=1)
    
    def create_bottom_panel(self):
        """Create bottom panel with progress and logs"""
        bottom_layout = QVBoxLayout()
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        bottom_layout.addLayout(progress_layout)
        
        # Logs
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(150)
        self.log_display.setFont(QFont("Courier New", 10))
        bottom_layout.addWidget(self.log_display)
        
        self.layout.addLayout(bottom_layout)
    
    def load_agents(self):
        """Load available agents"""
        # Clear existing items
        self.agent_selector.clear()
        
        # Get active agents
        active_agents = self.agent_manager.get_active_agents()
        
        # Filter to fine-tuning agents
        fine_tuning_agents = [
            agent for agent in active_agents
            if agent["agent_type"] == "fine_tuning"
        ]
        
        if not fine_tuning_agents:
            self.agent_selector.addItem("No fine-tuning agents available")
            self.train_button.setEnabled(False)
            self.generate_button.setEnabled(False)
            return
        
        # Add to combo box
        for agent in fine_tuning_agents:
            self.agent_selector.addItem(agent["agent_id"])
        
        # Select first agent
        if self.agent_selector.count() > 0:
            self.agent_selector.setCurrentIndex(0)
    
    def on_agent_selected(self, agent_id: str):
        """Handle agent selection
        
        Args:
            agent_id: ID of the selected agent
        """
        if agent_id == "No fine-tuning agents available":
            self.current_agent_id = None
            self.train_button.setEnabled(False)
            self.generate_button.setEnabled(False)
            return
        
        self.current_agent_id = agent_id
        self.train_button.setEnabled(True)
        self.generate_button.setEnabled(True)
    
    def create_new_agent(self):
        """Create a new fine-tuning agent"""
        # Find the main window
        main_window = self
        while main_window and not hasattr(main_window, 'create_fine_tuning_agent'):
            main_window = main_window.parent()
        
        if main_window and hasattr(main_window, 'create_fine_tuning_agent'):
            main_window.create_fine_tuning_agent()
        else:
            # Fallback if we can't find the method
            QMessageBox.warning(
                self,
                "Not Implemented",
                "Create fine-tuning agent functionality not found in main window."
            )
    
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir_edit.text()
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def add_example_row(self):
        """Add example instruction/output row to the dataset"""
        # Get current row count
        row_position = self.dataset_table.table.rowCount()
        self.dataset_table.table.insertRow(row_position)
        
        # Example data
        instruction = "Explain the concept of reinforcement learning"
        input_text = ""
        output = "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. The agent learns from the consequences of its actions, rather than from being explicitly taught, and it selects its actions based on its past experiences (exploitation) and also by new choices (exploration)."
        
        # Set data in table
        self.dataset_table.table.setItem(row_position, 0, QTableWidgetItem(instruction))
        self.dataset_table.table.setItem(row_position, 1, QTableWidgetItem(input_text))
        self.dataset_table.table.setItem(row_position, 2, QTableWidgetItem(output))
    
    def log_message(self, message: str):
        """Add message to log display
        
        Args:
            message: Message to log
        """
        self.log_display.append(message)
        
        # Scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def start_fine_tuning(self):
        """Start fine-tuning process"""
        if not self.current_agent_id:
            QMessageBox.warning(
                self,
                "Agent Required",
                "Please select or create a fine-tuning agent first."
            )
            return
        
        # Get dataset
        dataset = self.dataset_table.get_dataset()
        
        if not dataset:
            QMessageBox.warning(
                self,
                "Empty Dataset",
                "Please add at least one instruction/output pair to the dataset."
            )
            return
        
        # Get configuration
        config = {
            "model_id": self.model_selector.currentText(),
            "num_train_epochs": self.epochs_spin.value(),
            "per_device_train_batch_size": self.batch_size_spin.value(),
            "per_device_eval_batch_size": self.batch_size_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "lora_r": self.lora_r_spin.value(),
            "lora_alpha": self.lora_alpha_spin.value(),
            "output_dir": self.output_dir_edit.text(),
            "push_to_hub": self.push_to_hub_check.isChecked(),
            "hub_model_id": self.hub_model_id_edit.text() if self.push_to_hub_check.isChecked() else None
        }
        
        # Create command
        command = {
            "action": "fine_tune",
            "dataset": dataset,
            "test_size": 0.2,
            "config": config
        }
        
        # Convert to JSON
        input_text = json.dumps(command)
        
        # Clear log display
        self.log_display.clear()
        self.log_message("Starting fine-tuning process...")
        self.log_message(f"Model: {config['model_id']}")
        self.log_message(f"Dataset size: {len(dataset)} examples")
        self.log_message(f"Epochs: {config['num_train_epochs']}")
        self.log_message("Initializing...")
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
        # Disable training button
        self.train_button.setEnabled(False)
        
        # Find the main window
        main_window = self
        while main_window and not hasattr(main_window, 'run_agent_in_thread'):
            main_window = main_window.parent()
        
        if main_window and hasattr(main_window, 'run_agent_in_thread'):
            # Run agent in thread
            main_window.run_agent_in_thread(
                self.current_agent_id, 
                input_text,
                self.handle_fine_tuning_result
            )
        else:
            # Fallback if we can't find the method
            QMessageBox.warning(
                self,
                "Not Implemented",
                "run_agent_in_thread functionality not found in main window."
            )
            self.train_button.setEnabled(True)
    
    def handle_fine_tuning_result(self, result: str):
        """Handle fine-tuning result
        
        Args:
            result: Result from the agent
        """
        self.log_message("Fine-tuning process completed.")
        self.log_message(result)
        
        # Set progress bar to 100%
        self.progress_bar.setValue(100)
        
        # Enable training button
        self.train_button.setEnabled(True)
        
        # Show result
        try:
            # Parse result as JSON
            result_obj = json.loads(result)
            
            if result_obj.get("status") == "success":
                # Show success message
                QMessageBox.information(
                    self,
                    "Fine-Tuning Complete",
                    f"Fine-tuning completed successfully.\nModel saved to: {result_obj.get('model_path')}"
                )
                
                # Switch to test tab
                self.tabs.setCurrentIndex(2)
            else:
                # Show error message
                QMessageBox.warning(
                    self,
                    "Fine-Tuning Error",
                    f"Error during fine-tuning: {result_obj.get('message', 'Unknown error')}"
                )
        except json.JSONDecodeError:
            # Not valid JSON, show raw result
            QMessageBox.information(
                self,
                "Fine-Tuning Complete",
                "Fine-tuning process completed."
            )
    
    def generate_response(self):
        """Generate response from fine-tuned model"""
        if not self.current_agent_id:
            QMessageBox.warning(
                self,
                "Agent Required",
                "Please select or create a fine-tuning agent first."
            )
            return
        
        # Get instruction and input
        instruction = self.test_instruction_edit.toPlainText().strip()
        input_text = self.test_input_edit.toPlainText().strip()
        
        if not instruction:
            QMessageBox.warning(
                self,
                "Empty Instruction",
                "Please enter an instruction to test the model."
            )
            return
        
        # Create command
        command = {
            "action": "generate",
            "instruction": instruction,
            "input": input_text
        }
        
        # Convert to JSON
        input_json = json.dumps(command)
        
        # Clear output display
        self.test_output_display.clear()
        self.test_output_display.setPlaceholderText("Generating response...")
        
        # Disable generate button
        self.generate_button.setEnabled(False)
        
        # Find the main window
        main_window = self
        while main_window and not hasattr(main_window, 'run_agent_in_thread'):
            main_window = main_window.parent()
        
        if main_window and hasattr(main_window, 'run_agent_in_thread'):
            # Run agent in thread
            main_window.run_agent_in_thread(
                self.current_agent_id, 
                input_json,
                self.handle_generation_result
            )
        else:
            # Fallback if we can't find the method
            QMessageBox.warning(
                self,
                "Not Implemented",
                "run_agent_in_thread functionality not found in main window."
            )
            self.generate_button.setEnabled(True)
    
    def handle_generation_result(self, result: str):
        """Handle generation result
        
        Args:
            result: Result from the agent
        """
        # Display result
        self.test_output_display.setPlainText(result)
        
        # Enable generate button
        self.generate_button.setEnabled(True)