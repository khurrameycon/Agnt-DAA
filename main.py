#!/usr/bin/env python
"""
sagax1 - An Opensource AI-powered agent platform for everyday tasks
Main application entry point
"""

import sys
import os
import logging
import tempfile
from dotenv import load_dotenv
from app.utils.license_manager import LicenseManager
from app.ui.license_dialog import LicenseDialog
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog


# Fix for MultiplexedPath issue - MUST be before any imports that might use transformers
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Create a temporary directory for transformers cache
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(tempfile.gettempdir(), 'transformers_cache')
    os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
    
    # Set other environment variables that might help
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = 'False'
    os.environ['SAFETENSORS_FAST_GPU'] = '0'
    
    # Create a directory for gradio client cache
    os.environ['GRADIO_TEMP_DIR'] = os.path.join(tempfile.gettempdir(), 'gradio_cache')
    os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

# Now it's safe to import the rest
from app.core.config_manager import ConfigManager
from app.core.agent_manager import AgentManager
from app.ui.main_window import MainWindow
from app.utils.logging_utils import setup_logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QCoreApplication


#!/usr/bin/env python
"""
sagax1 - An Opensource AI-powered agent platform for everyday tasks
Main application entry point
"""

import sys
import os
import logging
from dotenv import load_dotenv
from app.core.config_manager import ConfigManager
from app.core.agent_manager import AgentManager
from app.ui.main_window import MainWindow
from app.utils.logging_utils import setup_logging
from app.utils.license_manager import LicenseManager
from app.ui.license_dialog import LicenseDialog
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QCoreApplication

def setup_environment():
    """Set up the environment variables and paths"""
    # Set application info for QSettings
    QCoreApplication.setOrganizationName("sagax1")
    QCoreApplication.setOrganizationDomain("sagax1.ai")
    QCoreApplication.setApplicationName("sagax1")
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Create necessary directories if they don't exist
    os.makedirs('config', exist_ok=True)
    os.makedirs('assets/icons', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def check_license(logger):
    """Check for a valid license and prompt for one if needed
    
    Args:
        logger: Logger instance
    
    Returns:
        bool: True if licensed, False otherwise
    """
    # Initialize license manager
    license_manager = LicenseManager()
    
    # Check if we already have a valid license
    if license_manager.is_licensed():
        logger.info("Valid license found")
        return True
    
    # Need to prompt for license
    logger.info("No valid license found, showing license dialog")
    
    # Create and show the license dialog
    dialog = LicenseDialog(license_manager)
    result = dialog.exec()
    
    # Check if license was successfully activated
    if result == QDialog.DialogCode.Accepted:
        logger.info("License activated successfully")
        return True
    else:
        logger.warning("User canceled license activation")
        return False

def main():
    """Main application entry point"""
    # Set up environment
    setup_environment()
    
    # Set up logging
    logger = setup_logging(log_level=logging.INFO)
    
    try:
        # Enable high DPI scaling
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("sagax1")
        app.setApplicationVersion("0.1.0")
        
        # Now we can import and use UI-related modules
        from app.utils.ui_assets import UIAssets
        from app.utils.style_system import StyleSystem
        from app.ui.splash_screen import sagax1SplashScreen
        
        # Create default icons now that QApplication exists
        UIAssets.create_default_icons_file()
        
        # Apply application icon
        UIAssets.apply_app_icon(app)
        
        # Apply stylesheet
        StyleSystem.apply_stylesheet(app)
        
        # Show splash screen
        splash = sagax1SplashScreen()
        splash.show()
        app.processEvents()
        
        # Check for valid license
        if not check_license(logger):
            logger.error("License check failed, exiting application")
            QMessageBox.critical(
                None,
                "License Required",
                "A valid license is required to use sagax1.\n"
                "Please purchase a license from www.yourdomain.com and try again."
            )
            return 1
        
        # Initialize configuration
        config_manager = ConfigManager()
        
        # Initialize agent manager
        agent_manager = AgentManager(config_manager)
        
        # Create main window
        window = MainWindow(agent_manager, config_manager)
        
        # Finish splash and show window
        splash.finish(window)
        window.show()
        
        logger.info("Application started")
        sys.exit(app.exec())
    
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        raise

if __name__ == "__main__":
    sys.exit(main())