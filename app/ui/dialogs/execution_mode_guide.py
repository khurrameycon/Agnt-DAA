"""
Execution Mode Guide for sagax1
Provides a dialog to explain differences between local and API execution
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class ExecutionModeGuideDialog(QDialog):
    """Dialog explaining differences between local and API execution modes"""
    
    def __init__(self, parent=None):
        """Initialize the dialog
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.setWindowTitle("Model Execution Mode Guide")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Choosing the Right Model Execution Mode")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Introduction
        intro = QLabel(
            "sagax1 offers two ways to run AI models. Choose the one that best fits your needs:"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        
        # Comparison table
        table = QTableWidget(5, 3)
        table.setHorizontalHeaderLabels(["Feature", "Local Execution", "Hugging Face API"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Fill comparison table
        comparisons = [
            ["Disk Space", "High (1-30GB per model)", "Minimal"],
            ["Memory Usage", "High (2-16GB RAM)", "Minimal"],
            ["Internet Required", "Only for download", "Always"],
            ["API Key", "Not required", "Required"],
            ["Speed", "Fast after loading", "Depends on internet"]
        ]
        
        for row, (feature, local, api) in enumerate(comparisons):
            table.setItem(row, 0, QTableWidgetItem(feature))
            table.setItem(row, 1, QTableWidgetItem(local))
            table.setItem(row, 2, QTableWidgetItem(api))
        
        layout.addWidget(table)
        
        # Recommendations
        recommendations = QLabel(
            "<b>Recommendations:</b><br>"
            "• <b>Local Execution:</b> Best for regular use on a powerful computer with good GPU/RAM<br>"
            "• <b>Hugging Face API:</b> Best for occasional use or on computers with limited resources<br>"
            "<br>"
            "<b>Note:</b> Some large models like Llama 3 (70B) are too big for most home computers "
            "and are best used via API."
        )
        recommendations.setWordWrap(True)
        recommendations.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(recommendations)
        
        # API key reminder
        api_reminder = QLabel(
            "Remember to set your Hugging Face API key in Settings if using API mode."
        )
        api_reminder.setStyleSheet("color: #FF6700; background-color: #FFEFDB; padding: 8px; border-radius: 4px;")
        api_reminder.setWordWrap(True)
        layout.addWidget(api_reminder)
        
        # OK button
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.setFixedWidth(100)
        ok_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)