"""
Conversation component for SagaX1
Handles displaying and managing conversation history
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QHBoxLayout,
    QPushButton, QLabel, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
import json
import logging
from typing import List, Dict, Any

class ConversationWidget(QWidget):
    """Widget for displaying and managing conversation history"""
    
    message_sent = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the conversation widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create conversation history
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        layout.addWidget(self.history)
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self.clear_history)
        button_layout.addWidget(self.clear_button)
        
        self.save_button = QPushButton("Save History")
        self.save_button.clicked.connect(self.save_history)
        button_layout.addWidget(self.save_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def add_message(self, message: str, is_user: bool = True):
        """Add a message to the conversation
        
        Args:
            message: Message text
            is_user: Whether the message is from the user
        """
        sender = "You" if is_user else "Agent"
        self.history.append(f"<b>{sender}:</b> {message}")
        
        # Scroll to bottom
        scroll_bar = self.history.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
    
    def clear_history(self):
        """Clear the conversation history"""
        self.history.clear()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history
        
        Returns:
            List of conversation messages
        """
        # This is a simple implementation that would need to be enhanced
        # to properly extract messages from HTML
        html = self.history.toHtml()
        
        # For now, just return a placeholder
        return [{"role": "system", "content": "History retrieval not fully implemented"}]
    
    def save_history(self):
        """Save the conversation history to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Conversation History",
            "",
            "Text Files (*.txt);;HTML Files (*.html);;JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith(".txt"):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.history.toPlainText())
            elif file_path.endswith(".html"):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.history.toHtml())
            elif file_path.endswith(".json"):
                history = self.get_history()
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
            
            self.logger.info(f"Saved conversation history to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {str(e)}")