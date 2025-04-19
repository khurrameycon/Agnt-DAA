"""
RAG Tab Component for sagax1
Tab for document retrieval and question answering interactions
"""

import os
import logging
from typing import Optional, Dict, Any, List
import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QFileDialog, QProgressBar,
    QMessageBox, QGroupBox, QFormLayout, QListWidget, QListWidgetItem,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QIcon

from app.core.agent_manager import AgentManager
from app.ui.components.conversation import ConversationWidget


class RagTab(QWidget):
    """Tab for document retrieval and question answering"""
    
    def __init__(self, agent_manager: AgentManager, parent=None):
        """Initialize the RAG tab
        
        Args:
            agent_manager: Agent manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.agent_manager = agent_manager
        self.logger = logging.getLogger(__name__)
        self.current_agent_id = None
        self.current_document_id = None
        self.uploaded_documents = {}  # Dictionary of document ID to document info
        
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
        top_layout.addWidget(QLabel("Select RAG Agent:"))
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
        """Create main panel with document list, document view, and conversation"""
        # Create splitter for content and conversation
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top panel for document management
        doc_panel = QWidget()
        doc_layout = QVBoxLayout(doc_panel)
        
        # Document controls
        doc_controls = QHBoxLayout()
        
        # Upload button
        self.upload_button = QPushButton("Upload Document")
        self.upload_button.clicked.connect(self.upload_document)
        self.upload_button.setEnabled(False)
        doc_controls.addWidget(self.upload_button)
        
        # Clear documents button
        self.clear_docs_button = QPushButton("Clear Documents")
        self.clear_docs_button.clicked.connect(self.clear_documents)
        self.clear_docs_button.setEnabled(False)
        doc_controls.addWidget(self.clear_docs_button)
        
        doc_controls.addStretch()
        doc_layout.addLayout(doc_controls)
        
        # Create splitter for document list and document view
        doc_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Document list
        doc_list_group = QGroupBox("Uploaded Documents")
        doc_list_layout = QVBoxLayout(doc_list_group)
        
        self.doc_list = QListWidget()
        self.doc_list.currentItemChanged.connect(self.on_document_selected)
        doc_list_layout.addWidget(self.doc_list)
        
        doc_splitter.addWidget(doc_list_group)
        
        # Document info panel
        doc_info_group = QGroupBox("Document Information")
        doc_info_layout = QVBoxLayout(doc_info_group)
        
        self.doc_info = QTextEdit()
        self.doc_info.setReadOnly(True)
        self.doc_info.setPlaceholderText("Select a document to view information")
        doc_info_layout.addWidget(self.doc_info)
        
        doc_splitter.addWidget(doc_info_group)
        
        # Set initial sizes
        doc_splitter.setSizes([200, 400])
        
        doc_layout.addWidget(doc_splitter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        doc_layout.addWidget(self.progress_bar)
        
        # Add to main splitter
        splitter.addWidget(doc_panel)
        
        # Conversation widget
        self.conversation = ConversationWidget()
        splitter.addWidget(self.conversation)
        
        # Set initial sizes
        splitter.setSizes([300, 400])
        
        # Add splitter to layout
        self.layout.addWidget(splitter, stretch=1)
    
    def create_input_panel(self):
        """Create input panel for queries"""
        input_layout = QHBoxLayout()
        
        # Query input
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Ask a question about the uploaded document...")
        self.query_input.setMaximumHeight(100)
        input_layout.addWidget(self.query_input)
        
        # Send button
        self.send_button = QPushButton("Send Query")
        self.send_button.clicked.connect(self.send_query)
        self.send_button.setEnabled(False)
        input_layout.addWidget(self.send_button)
        
        self.layout.addLayout(input_layout)
    
    def load_agents(self):
        """Load available agents"""
        # Clear existing items
        self.agent_selector.clear()
        
        # Get active agents
        active_agents = self.agent_manager.get_active_agents()
        
        # Filter to RAG agents
        rag_agents = [
            agent for agent in active_agents
            if agent["agent_type"] == "rag"
        ]
        
        if not rag_agents:
            self.agent_selector.addItem("No RAG agents available")
            self.upload_button.setEnabled(False)
            self.send_button.setEnabled(False)
            self.clear_docs_button.setEnabled(False)
            return
        
        # Add to combo box
        for agent in rag_agents:
            self.agent_selector.addItem(agent["agent_id"])
        
        # Select first agent
        if self.agent_selector.count() > 0:
            self.agent_selector.setCurrentIndex(0)
    
    def on_agent_selected(self, agent_id: str):
        """Handle agent selection
        
        Args:
            agent_id: ID of the selected agent
        """
        if agent_id == "No RAG agents available":
            self.current_agent_id = None
            self.upload_button.setEnabled(False)
            self.send_button.setEnabled(False)
            self.clear_docs_button.setEnabled(False)
            return
        
        try:
            # Update current agent
            self.current_agent_id = agent_id
            
            # Enable upload button
            self.upload_button.setEnabled(True)
            
            # Clear document list
            self.clear_document_list()
            
            # Disable send button until document is uploaded
            self.send_button.setEnabled(False)
            self.clear_docs_button.setEnabled(False)
            
            # Show agent selected message
            self.conversation.add_message(f"Agent '{agent_id}' selected. Please upload a document.", is_user=False)
        except Exception as e:
            self.logger.error(f"Error selecting agent: {str(e)}")
            self.current_agent_id = None
            self.upload_button.setEnabled(False)
            self.send_button.setEnabled(False)
            self.clear_docs_button.setEnabled(False)
    
    def create_new_agent(self):
        """Create a new RAG agent"""
        # Find the main window by traversing up the parent hierarchy
        main_window = self
        while main_window and not hasattr(main_window, 'create_rag_agent'):
            main_window = main_window.parent()
        
        if main_window and hasattr(main_window, 'create_rag_agent'):
            main_window.create_rag_agent()
        else:
            # Fallback if we can't find the method
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Not Implemented",
                "Create RAG agent functionality not found in main window."
            )
    
    def upload_document(self):
        """Upload a document"""
        if not self.current_agent_id:
            QMessageBox.warning(self, "No Agent Selected", "Please select an agent first.")
            return
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            "PDF Files (*.pdf);;Text Files (*.txt);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.upload_button.setEnabled(False)
        
        # Add to conversation
        self.conversation.add_message(f"Uploading document: {os.path.basename(file_path)}", is_user=True)
        
        # Find the main window to run agent in thread
        main_window = self.window()
        if hasattr(main_window, 'run_agent_in_thread'):
            # Create upload command
            command = {
                "action": "upload",
                "file_path": file_path
            }
            
            # Convert to JSON
            input_json = json.dumps(command)
            
            # Execute the agent in a thread
            main_window.run_agent_in_thread(
                self.current_agent_id, 
                input_json,
                self.handle_upload_result
            )
        else:
            self.logger.error("Main window not found or missing run_agent_in_thread method")
            self.conversation.add_message("Error: Could not upload document. Please try again.", is_user=False)
            
            # Hide progress bar
            self.progress_bar.setVisible(False)
            
            # Enable upload button
            self.upload_button.setEnabled(True)
    
    def handle_upload_result(self, result: str):
        """Handle upload result
        
        Args:
            result: Upload result from agent
        """
        try:
            # Parse result as JSON
            result_data = json.loads(result)
            
            if result_data.get("success", False):
                # Add document to list
                document_id = result_data.get("document_id")
                file_name = result_data.get("file_name")
                
                # Store document info
                self.uploaded_documents[document_id] = result_data
                
                # Add document to list
                item = QListWidgetItem(file_name)
                item.setData(Qt.ItemDataRole.UserRole, document_id)
                self.doc_list.addItem(item)
                
                # Select the newly added document
                self.doc_list.setCurrentItem(item)
                
                # Enable send button
                self.send_button.setEnabled(True)
                self.clear_docs_button.setEnabled(True)
                
                # Add success message to conversation
                self.conversation.add_message(result_data.get("message", "Document uploaded successfully."), is_user=False)
            else:
                # Add error message to conversation
                error_msg = result_data.get("error", "Unknown error")
                self.conversation.add_message(f"Error uploading document: {error_msg}", is_user=False)
        except json.JSONDecodeError:
            # Not JSON, just add the raw result
            self.conversation.add_message(result, is_user=False)
        except Exception as e:
            # Add error message to conversation
            self.conversation.add_message(f"Error handling upload result: {str(e)}", is_user=False)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Enable upload button
        self.upload_button.setEnabled(True)
    
    def clear_document_list(self):
        """Clear the document list"""
        self.doc_list.clear()
        self.doc_info.clear()
        self.uploaded_documents = {}
        self.current_document_id = None
    
    def clear_documents(self):
        """Clear all documents"""
        self.clear_document_list()
        self.send_button.setEnabled(False)
        self.clear_docs_button.setEnabled(False)
        
        # Add message to conversation
        self.conversation.add_message("All documents cleared.", is_user=False)
    
    def on_document_selected(self, current, previous):
        """Handle document selection
        
        Args:
            current: Current selected item
            previous: Previously selected item
        """
        if not current:
            self.doc_info.clear()
            self.current_document_id = None
            return
        
        # Get document ID
        document_id = current.data(Qt.ItemDataRole.UserRole)
        self.current_document_id = document_id
        
        # Get document info
        document_info = self.uploaded_documents.get(document_id, {})
        
        # Display document info
        if document_info:
            # Format document info
            info_text = f"Document ID: {document_id}\n"
            info_text += f"File Name: {document_info.get('file_name', 'Unknown')}\n"
            
            # Add additional info if available
            if "raw_result" in document_info:
                raw_result = document_info["raw_result"]
                if isinstance(raw_result, dict):
                    for key, value in raw_result.items():
                        if key not in ["file_path"]:  # Skip file path
                            info_text += f"{key}: {value}\n"
                else:
                    info_text += f"Raw Result: {raw_result}\n"
            
            self.doc_info.setText(info_text)
        else:
            self.doc_info.setText(f"No information available for document ID: {document_id}")
    
    def send_query(self):
        """Send a query to the agent"""
        if not self.current_agent_id:
            QMessageBox.warning(self, "No Agent Selected", "Please select an agent first.")
            return
        
        if not self.current_document_id:
            QMessageBox.warning(self, "No Document Selected", "Please upload and select a document first.")
            return
        
        # Get query
        query = self.query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a query.")
            return
        
        # Add to conversation
        self.conversation.add_message(query, is_user=True)
        
        # Clear input
        self.query_input.clear()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        
        # Disable input
        self.query_input.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Find the main window to run agent in thread
        main_window = self.window()
        if hasattr(main_window, 'run_agent_in_thread'):
            # Create query command
            command = {
                "action": "query",
                "query": query,
                "document_id": self.current_document_id
            }
            
            # Convert to JSON
            input_json = json.dumps(command)
            
            # Execute the agent in a thread
            main_window.run_agent_in_thread(
                self.current_agent_id, 
                input_json,
                self.handle_query_result
            )
        else:
            self.logger.error("Main window not found or missing run_agent_in_thread method")
            self.conversation.add_message("Error: Could not process query. Please try again.", is_user=False)
            
            # Hide progress bar
            self.progress_bar.setVisible(False)
            
            # Enable input
            self.query_input.setEnabled(True)
            self.send_button.setEnabled(True)
    
    def handle_query_result(self, result: str):
        """Handle query result
        
        Args:
            result: Query result from agent
        """
        try:
            # Parse result as JSON
            try:
                result_data = json.loads(result)
                
                if result_data.get("success", False):
                    # Add answer to conversation
                    answer = result_data.get("answer", "No answer provided.")
                    self.conversation.add_message(answer, is_user=False)
                else:
                    # Add error message to conversation
                    error_msg = result_data.get("error", "Unknown error")
                    self.conversation.add_message(f"Error processing query: {error_msg}", is_user=False)
            except json.JSONDecodeError:
                # Not JSON, just add the raw result
                self.conversation.add_message(result, is_user=False)
        except Exception as e:
            # Add error message to conversation
            self.conversation.add_message(f"Error handling query result: {str(e)}", is_user=False)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Enable input
        self.query_input.setEnabled(True)
        self.send_button.setEnabled(True)
    
    def update_progress(self, progress_text: str):
        """Update progress bar based on progress text
        
        Args:
            progress_text: Progress text
        """
        if "uploading" in progress_text.lower():
            self.progress_bar.setValue(30)
        elif "processing" in progress_text.lower():
            self.progress_bar.setValue(60)
        elif "success" in progress_text.lower():
            self.progress_bar.setValue(100)
        else:
            # Increment by 10%
            current_value = self.progress_bar.value()
            self.progress_bar.setValue(min(current_value + 10, 90))