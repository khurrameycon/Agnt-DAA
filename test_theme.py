import sys
import os
import traceback
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
    QTextEdit, QMessageBox, QHBoxLayout, QPushButton
)
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt  # Corrected import for Qt

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from app.ui.theme_manager import ThemeManager
from app.ui.font_manager import FontManager

def show_error_dialog(error_message):
    """Display an error dialog with detailed information"""
    error_dialog = QMessageBox()
    error_dialog.setIcon(QMessageBox.Icon.Critical)
    error_dialog.setWindowTitle("Theme Application Error")
    error_dialog.setText("An error occurred while applying the theme:")
    error_dialog.setDetailedText(error_message)
    error_dialog.exec()

def test_theme():
    """Test theme and font loading"""
    try:
        # Create application
        app = QApplication(sys.argv)
        
        print("--- Font Loading Test ---")
        # Load fonts
        loaded_fonts = FontManager.load_custom_fonts()
        print(f"Loaded font IDs: {loaded_fonts}")
        
        # Get font families
        font_families = FontManager.get_font_families()
        print(f"Available font families: {font_families}")
        
        # Get primary font
        primary_font = FontManager.get_primary_font()
        print(f"Primary font: {primary_font}")
        
        print("\n--- Theme Application Test ---")
        # Apply theme
        ThemeManager.apply_dark_theme(app)
        
        # Create main window
        window = QMainWindow()
        window.setWindowTitle("SagaX1 Theme Test")
        
        # Create central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Title Label
        title_label = QLabel("SagaX1 Theme Test")
        title_label.setStyleSheet("""
            font-size: 24px; 
            color: #3584E4; 
            font-weight: bold;
        """)
        layout.addWidget(title_label)
        
        # Description Label
        description_label = QLabel(f"Primary Font: {primary_font}")
        description_label.setStyleSheet("""
            font-size: 16px;
            color: #FFFFFF;
        """)
        layout.addWidget(description_label)
        
        # Text Edit to show styling
        text_edit = QTextEdit()
        text_edit.setPlaceholderText("Enter text here to test styling...")
        layout.addWidget(text_edit)
        
        # Buttons to show interaction
        button_layout = QHBoxLayout()
        
        # Confirm Button
        confirm_button = QPushButton("Confirm")
        confirm_button.setStyleSheet("""
            background-color: #3584E4; 
            color: white; 
            padding: 10px;
            border-radius: 5px;
        """)
        confirm_button.setFixedWidth(100)
        button_layout.addWidget(confirm_button)
        
        # Cancel Button
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("""
            background-color: #FF4136; 
            color: white; 
            padding: 10px;
            border-radius: 5px;
        """)
        cancel_button.setFixedWidth(100)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        window.setCentralWidget(central_widget)
        window.resize(500, 400)
        window.show()
        
        sys.exit(app.exec())
    
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        
        # Create a separate application to show error dialog
        error_app = QApplication([])
        show_error_dialog(str(traceback.format_exc()))
        sys.exit(error_app.exec())

if __name__ == "__main__":
    test_theme()