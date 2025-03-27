#!/usr/bin/env python
"""
sagax1 - An AI-powered agent platform for everyday tasks
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

def setup_environment():
    """Set up the environment variables and paths"""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Create necessary directories if they don't exist
    os.makedirs('config', exist_ok=True)
    os.makedirs('assets/icons', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def main():
    """Main application entry point"""
    # Set up environment
    setup_environment()
    
    # Set up logging
    logger = setup_logging(log_level=logging.INFO)
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        
        # Initialize agent manager
        agent_manager = AgentManager(config_manager)
        
        # Create and show UI
        app = QApplication(sys.argv)
        app.setApplicationName("sagax1")
        app.setApplicationVersion("0.1.0")
        
        window = MainWindow(agent_manager, config_manager)
        window.show()
        
        logger.info("Application started")
        sys.exit(app.exec())
    
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        raise

if __name__ == "__main__":
    main()