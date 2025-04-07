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
from PyQt6.QtWidgets import QApplication
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
    
    # Note: We don't call UIAssets.ensure_assets_exist() here to avoid QPixmap issues

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
        
        # Initialize configuration
        config_manager = ConfigManager()
        
        # Initialize agent manager
        agent_manager = AgentManager(config_manager)
        
        # Create main window
        window = MainWindow(agent_manager, config_manager)
        
        # Show splash screen
        splash = sagax1SplashScreen()
        splash.show_with_timer(app, window, 2500)  # Show splash for 2.5 seconds
        
        logger.info("Application started")
        sys.exit(app.exec())
    
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        raise

if __name__ == "__main__":
    main()