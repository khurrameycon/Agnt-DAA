"""
License Dialog for sagax1
Dialog for entering and validating license key
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class LicenseDialog(QDialog):
    """Dialog for license validation"""
    
    def __init__(self, license_manager, parent=None):
        """Initialize the license dialog
        
        Args:
            license_manager: License manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.license_manager = license_manager
        
        self.setWindowTitle("sagax1 License Activation")
        self.setMinimumWidth(450)
        self.setMinimumHeight(200)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # Add header with instructions
        header_label = QLabel("License Activation")
        header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.layout.addWidget(header_label)
        
        instructions = QLabel(
            "Please enter your license key from SendOwl to activate sagax1. "
            "You only need to do this once."
        )
        instructions.setWordWrap(True)
        self.layout.addWidget(instructions)
        
        # License key input
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("License Key:"))
        
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter your license key")
        key_layout.addWidget(self.key_input)
        
        self.layout.addLayout(key_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.activate_button = QPushButton("Activate")
        self.activate_button.clicked.connect(self.activate_license)
        
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.activate_button)
        button_layout.addWidget(self.exit_button)
        
        self.layout.addLayout(button_layout)
        
        # License status
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.layout.addWidget(self.status_label)
        
        # Set focus to key input
        self.key_input.setFocus()
    
    def activate_license(self):
        """Validate and activate the license"""
        license_key = self.key_input.text().strip()
        
        if not license_key:
            self.status_label.setText("Please enter a license key")
            self.status_label.setStyleSheet("color: red")
            return
        
        # Disable UI during validation
        self.setEnabled(False)
        QApplication.processEvents()
        
        # Validate the license
        success, message = self.license_manager.validate_license(license_key)
        
        # Re-enable UI
        self.setEnabled(True)
        
        if success:
            self.status_label.setText(f"Success: {message}")
            self.status_label.setStyleSheet("color: green")
            
            # Show success message and accept dialog
            QMessageBox.information(
                self,
                "License Activated",
                "Your license has been successfully activated.\n"
                "Thank you for using sagax1!"
            )
            
            self.accept()
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: red")
            
            # Clear the input for retry
            self.key_input.clear()
            self.key_input.setFocus()
            
            QMessageBox.warning(
                self,
                "License Activation Failed",
                f"Failed to activate license: {message}\n\n"
                "Please check your license key and try again."
            )