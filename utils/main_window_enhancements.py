"""
Main Window Enhancements for sagax1
Implementation file for enhancing the MainWindow class
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QComboBox,
    QStatusBar, QTabWidget, QLineEdit, QMessageBox,
    QMenuBar, QMenu, QDialog, QDialogButtonBox, QFormLayout,
    QListWidget, QListWidgetItem, QSplitter, QCheckBox,
    QToolBar, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QThread, pyqtSlot
from PyQt6.QtGui import QIcon, QAction, QFont, QPixmap

from app.utils.style_system import StyleSystem
from app.utils.ui_assets import UIAssets

def enhance_main_window(MainWindow):
    """
    Enhance the MainWindow class with professional UI elements and fix layout issues
    
    Args:
        MainWindow: The MainWindow class to enhance
    """
    # Store the original __init__ method
    original_init = MainWindow.__init__
    
    # Define the new enhanced __init__ method
    def enhanced_init(self, agent_manager, config_manager):
        # Call the original __init__ method
        original_init(self, agent_manager, config_manager)
        
        # Apply enhancements to this instance
        try:
            _apply_enhancements(self)
            
            # Fix layout issues - do this after a small delay
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self.adjustSize())
        except Exception as e:
            print(f"Error in enhanced_init: {str(e)}")
    
    # Replace the __init__ method with the enhanced one
    MainWindow.__init__ = enhanced_init
    
    # Store the original show_about method
    if hasattr(MainWindow, 'show_about'):
        original_show_about = MainWindow.show_about
        
        # Replace with enhanced version
        def enhanced_show_about(self):
            _show_enhanced_about(self)
        
        MainWindow.show_about = enhanced_show_about
    
    # Return the enhanced class
    return MainWindow

def _apply_enhancements(window):
    """Apply UI enhancements to the window instance
    
    Args:
        window: The MainWindow instance to enhance
    """
    try:
        # Set window properties for better appearance
        window.setWindowTitle("My1ai - My1 Agnostic Intelligence")
        window.setMinimumSize(1100, 800)
        
        # Apply the application icon
        window.setWindowIcon(UIAssets.get_icon("app"))
        
        # Create or update the toolbar to be more professional
        _create_toolbar(window)
        
        # Enhance the header area
        _create_header(window)
        
        # Style all send buttons with accent color
        _style_send_buttons(window)
        
        # Style all tab widgets
        _style_tab_widgets(window)
        
        # Make the status bar more informative and professional
        _enhance_status_bar(window)
    except Exception as e:
        print(f"Error in _apply_enhancements: {str(e)}")

def _create_toolbar(window):
    """Create or enhance the toolbar with professional icons
    
    Args:
        window: The MainWindow instance
    """
    try:
        # Create toolbar if it doesn't exist
        if not hasattr(window, 'toolbar'):
            window.toolbar = QToolBar("Main Toolbar")
            window.addToolBar(window.toolbar)
        else:
            window.toolbar.clear()
        
        # Set toolbar properties
        window.toolbar.setMovable(False)
        window.toolbar.setIconSize(QSize(0, 0))
        window.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # Add actions with professional icons
        # create_new_action = QAction(UIAssets.get_icon("create"), "New Agent", window)
        # create_new_action.triggered.connect(window.create_new_agent)
        # window.toolbar.addAction(create_new_action)
        
        # Add spacer
        # spacer = QWidget()
        # spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # window.toolbar.addWidget(spacer)
        
        # Add settings action on the right
        # settings_action = QAction(UIAssets.get_icon("settings"), "Settings", window)
        # settings_action.triggered.connect(lambda: window.tabs.setCurrentIndex(window.tabs.count() - 1))  # Go to settings tab
        # window.toolbar.addAction(settings_action)
    except Exception as e:
        print(f"Error in _create_toolbar: {str(e)}")

def _create_header(window):
    """Create a professional header area
    
    Args:
        window: The MainWindow instance
    """
    try:
        # Create header frame if it doesn't exist
        if not hasattr(window, 'header_frame'):
            # Create new header frame
            window.header_frame = QFrame()
            window.header_frame.setFrameShape(QFrame.Shape.StyledPanel)
            window.header_frame.setStyleSheet(f"background-color: {StyleSystem.COLORS['primary']}; color: white; border-radius: 0px;")
            
            # Create layout for header
            header_layout = QHBoxLayout(window.header_frame)
            header_layout.setContentsMargins(20, 10, 20, 10)
            
            # Add logo to header
            logo_label = QLabel()
            logo_pixmap = UIAssets.get_pixmap("app", 40, 40)
            if not logo_pixmap.isNull():
                logo_label.setPixmap(logo_pixmap)
            else:
                # Use text as fallback
                logo_label.setText("My1ai")
                logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
            
            header_layout.addWidget(logo_label)
            
            # Add title to header
            title_label = QLabel("My1ai")
            title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
            header_layout.addWidget(title_label)
            
            # Add tagline
            tagline_label = QLabel("My1 Agnostic Intelligence")
            tagline_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 14px;")
            header_layout.addSpacing(20)
            header_layout.addWidget(tagline_label)
            
            # Add stretch to push version to the right
            header_layout.addStretch()
            
            # Add version
            version_label = QLabel("v0.1.0")
            version_label.setStyleSheet("color: rgba(255, 255, 255, 0.5);")
            header_layout.addWidget(version_label)

            # Simpler approach - just create a toolbar and add our header frame to it
            header_toolbar = QToolBar("Header Toolbar")
            header_toolbar.setMovable(False)
            header_toolbar.setFloatable(False)
            header_toolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea)
            
            # Add the frame to a widget to add to the toolbar
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.addWidget(window.header_frame)
            
            # Add to toolbar
            header_toolbar.addWidget(container)
            
            # Add toolbar to window
            window.addToolBar(Qt.ToolBarArea.TopToolBarArea, header_toolbar)
    except Exception as e:
        print(f"Error creating header: {str(e)}")

def _style_send_buttons(window):
    """Style all send buttons to make them more attractive
    
    Args:
        window: The MainWindow instance
    """
    try:
        # Find all send buttons in the UI
        send_buttons = window.findChildren(QPushButton)
        
        for button in send_buttons:
            if hasattr(button, 'text') and button.text() in ["Send", "Generate", "Generate Code"]:
                # Style as action button
                StyleSystem.create_action_button(button)
                
                # Add icon if not already set
                if button.icon().isNull():
                    button.setIcon(UIAssets.get_icon("send"))
                
                # Set minimum width
                button.setMinimumWidth(120)
    except Exception as e:
        print(f"Error in _style_send_buttons: {str(e)}")

def _style_tab_widgets(window):
    """Style all tab widgets with icons for a more professional look
    
    Args:
        window: The MainWindow instance
    """
    try:
        # Set icons for the main tabs
        tab_icons = {
            "Chat": "chat",
            "Web Browsing": "web",
            "Visual Web": "visual",
            "Code Generation": "code",
            "Media Generation": "media",
            "Fine-Tuning": "fine_tuning",
            "RAG": "search",
            "Agents": "agents",
            "Settings": "settings"
        }
        
        # Apply icons to tabs
        for i in range(window.tabs.count()):
            tab_text = window.tabs.tabText(i)
            if tab_text in tab_icons:
                window.tabs.setTabIcon(i, UIAssets.get_icon(tab_icons[tab_text]))
    except Exception as e:
        print(f"Error in _style_tab_widgets: {str(e)}")

def _show_enhanced_about(window):
    """Show an enhanced about dialog
    
    Args:
        window: The MainWindow instance
    """
    try:
        about_dialog = QDialog(window)
        about_dialog.setWindowTitle("About My1ai")
        about_dialog.setMinimumWidth(480)
        about_dialog.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: bold;
                color: #2D5F8B;
            }
            QLabel#version {
                font-size: 14px;
                color: #666;
            }
            QLabel#description {
                font-size: 14px;
                color: #333;
            }
            QLabel#copyright {
                font-size: 12px;
                color: #999;
            }
        """)
        
        layout = QVBoxLayout(about_dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Logo
        logo_label = QLabel()
        logo_pixmap = UIAssets.get_pixmap("app", 80, 80)
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_label)
        
        # Title
        title_label = QLabel("My1ai")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Version
        version_label = QLabel("Version 0.1.0")
        version_label.setObjectName("version")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)
        
        # Description
        description_label = QLabel("My1 Agnostic Intelligence")
        description_label.setObjectName("description")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        
        # Copyright
        copyright_label = QLabel("© 2025 My1ai Team. All rights reserved.")
        copyright_label.setObjectName("copyright")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)
        
        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(about_dialog.accept)
        ok_button.setMinimumWidth(100)
        ok_button.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Layout for OK button (centered)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        about_dialog.exec()
    except Exception as e:
        print(f"Error in _show_enhanced_about: {str(e)}")

def _enhance_status_bar(window):
    """Enhance the status bar for a more professional look
    
    Args:
        window: The MainWindow instance
    """
    try:
        # Create a more informative status bar
        status_bar = window.statusBar()
        
        # Set a default message with application information
        status_bar.showMessage("My1ai - Ready")
        
        # Add permanent widgets to the right side
        api_status = QLabel()
        if window.config_manager.get_hf_api_key():
            api_status.setText("API: Connected ✓")
            api_status.setStyleSheet("color: white; background-color: #4CAF50; padding: 2px 8px; border-radius: 4px;")
        else:
            api_status.setText("API: Not Set ✗")
            api_status.setStyleSheet("color: white; background-color: #F44336; padding: 2px 8px; border-radius: 4px;")
        
        status_bar.addPermanentWidget(api_status)
        
        # Add models count
        models_count = len(window.agent_manager.model_manager.get_cached_models())
        models_label = QLabel(f"Models: {models_count}")
        models_label.setStyleSheet("color: white; margin-right: 10px;")
        status_bar.addPermanentWidget(models_label)
        
        # Store these widgets on the main window for later updating
        window.status_api = api_status
        window.status_models = models_label
    except Exception as e:
        print(f"Error in _enhance_status_bar: {str(e)}")