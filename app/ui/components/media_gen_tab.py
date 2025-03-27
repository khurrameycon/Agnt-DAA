"""
Media Generation Tab Component for SagaX1
Tab for media generation agent interaction
"""

import os
import logging
from typing import Optional
import tempfile

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QComboBox, QSplitter, QTabWidget,
    QRadioButton, QButtonGroup, QScrollArea, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QPixmap, QImage

from app.core.agent_manager import AgentManager
from app.ui.components.conversation import ConversationWidget


class MediaGenTab(QWidget):
    """Tab for media generation"""
    
    def __init__(self, agent_manager: AgentManager, parent=None):
        """Initialize the media generation tab
        
        Args:
            agent_manager: Agent manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.agent_manager = agent_manager
        self.logger = logging.getLogger(__name__)
        self.current_agent_id = None
        self.current_media_path = None
        self.media_type = "image"  # Default to image
        
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
        top_layout.addWidget(QLabel("Select Media Generation Agent:"))
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
        
        # Media type selection
        media_type_group = QButtonGroup(self)
        
        self.image_radio = QRadioButton("Image")
        self.image_radio.setChecked(True)
        self.image_radio.toggled.connect(self.on_media_type_changed)
        media_type_group.addButton(self.image_radio)
        top_layout.addWidget(self.image_radio)
        
        self.video_radio = QRadioButton("Video")
        self.video_radio.toggled.connect(self.on_media_type_changed)
        media_type_group.addButton(self.video_radio)
        top_layout.addWidget(self.video_radio)
        
        top_layout.addStretch()
        
        self.layout.addLayout(top_layout)
    
    def create_main_panel(self):
        """Create main panel with media display and conversation"""
        # Create splitter for media display and conversation
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Media panel
        media_panel = QWidget()
        media_layout = QVBoxLayout(media_panel)
        
        # Create scroll area for media display
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create label for media display
        self.media_display = QLabel("No media generated yet")
        self.media_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.media_display.setMinimumHeight(300)
        
        # Add to scroll area
        scroll_area.setWidget(self.media_display)
        media_layout.addWidget(scroll_area)
        
        # Add save button
        save_button = QPushButton("Save Media")
        save_button.clicked.connect(self.save_media)
        media_layout.addWidget(save_button)
        
        # Add to splitter
        splitter.addWidget(media_panel)
        
        # Conversation widget
        self.conversation = ConversationWidget()
        splitter.addWidget(self.conversation)
        
        # Add splitter to layout
        self.layout.addWidget(splitter, stretch=1)
    
    def create_input_panel(self):
        """Create input panel for user prompts"""
        input_layout = QHBoxLayout()
        
        # Prompt input
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt to generate an image or video...")
        self.prompt_input.setMaximumHeight(100)
        input_layout.addWidget(self.prompt_input)
        
        # Generate button
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_media)
        self.generate_button.setEnabled(False)
        input_layout.addWidget(self.generate_button)
        
        self.layout.addLayout(input_layout)
    
    def load_agents(self):
        """Load available agents"""
        # Clear existing items
        self.agent_selector.clear()
        
        # Get active agents
        active_agents = self.agent_manager.get_active_agents()
        
        # Filter to media generation agents
        media_gen_agents = [
            agent for agent in active_agents
            if agent["agent_type"] == "media_generation"
        ]
        
        if not media_gen_agents:
            self.agent_selector.addItem("No media generation agents available")
            self.generate_button.setEnabled(False)
            return
        
        # Add to combo box
        for agent in media_gen_agents:
            self.agent_selector.addItem(agent["agent_id"])
        
        # Select first agent
        if self.agent_selector.count() > 0:
            self.agent_selector.setCurrentIndex(0)
    
    def on_agent_selected(self, agent_id: str):
        """Handle agent selection
        
        Args:
            agent_id: ID of the selected agent
        """
        if agent_id == "No media generation agents available":
            self.current_agent_id = None
            self.generate_button.setEnabled(False)
            return
        
        try:
            # Update current agent
            self.current_agent_id = agent_id
            
            # Enable generate button
            self.generate_button.setEnabled(True)
        except Exception as e:
            self.logger.error(f"Error selecting agent: {str(e)}")
            self.current_agent_id = None
            self.generate_button.setEnabled(False)
    
    def on_media_type_changed(self, checked: bool):
        """Handle media type change
        
        Args:
            checked: Whether the radio button is checked
        """
        if not checked:
            return
        
        if self.sender() == self.image_radio:
            self.media_type = "image"
            self.prompt_input.setPlaceholderText("Enter your prompt to generate an image...")
        else:
            self.media_type = "video"
            self.prompt_input.setPlaceholderText("Enter your prompt to generate a video...")
    
    def create_new_agent(self):
        """Create a new media generation agent"""
        # Find the main window by traversing up the parent hierarchy
        main_window = self
        while main_window and not hasattr(main_window, 'create_media_generation_agent'):
            main_window = main_window.parent()
        
        if main_window and hasattr(main_window, 'create_media_generation_agent'):
            main_window.create_media_generation_agent()
        else:
            # Fallback if we can't find the method
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Not Implemented",
                "Create media generation agent functionality not found in main window."
            )
    
    def generate_media(self):
        """Generate media from prompt"""
        if self.current_agent_id is None:
            return
        
        # Get prompt
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            return
        
        # Add media type to prompt
        if self.media_type == "image":
            enhanced_prompt = f"Generate an image: {prompt}"
        else:
            enhanced_prompt = f"Generate a video: {prompt}"
        
        # Add to conversation
        self.conversation.add_message(enhanced_prompt, is_user=True)
        
        # Clear input
        self.prompt_input.clear()
        
        # Disable input
        self.prompt_input.setEnabled(False)
        self.generate_button.setEnabled(False)
        
        # Find the main window
        main_window = self
        while main_window and not hasattr(main_window, 'run_agent_in_thread'):
            main_window = main_window.parent()
        
        if main_window and hasattr(main_window, 'run_agent_in_thread'):
            # Run agent in thread
            main_window.run_agent_in_thread(
                self.current_agent_id, 
                enhanced_prompt,
                self.handle_agent_result
            )
        else:
            # Enable input again if we can't find the method
            self.prompt_input.setEnabled(True)
            self.generate_button.setEnabled(True)
            self.conversation.add_message("Error: Unable to run agent thread", is_user=False)
    
    def handle_agent_result(self, result: str):
        """Handle agent result
        
        Args:
            result: Agent result
        """
        # Add to conversation
        self.conversation.add_message(result, is_user=False)
        
        # Try to extract media path from result
        import re
        
        # Look for image or video file paths
        media_path_match = re.search(r"(?:saved to|generated at|created at|file:|path:)\s*([^\s]+\.(?:png|jpg|jpeg|gif|mp4|avi|mov|webm))", result, re.IGNORECASE)
        
        if media_path_match:
            media_path = media_path_match.group(1)
            self.display_media(media_path)
        else:
            # Look for temporary file paths
            temp_path_match = re.search(r"(/tmp/[^\s]+|C:\\Users\\[^\\]+\\AppData\\Local\\Temp\\[^\s]+)", result)
            if temp_path_match:
                media_path = temp_path_match.group(1)
                self.display_media(media_path)
        
        # Enable input
        self.prompt_input.setEnabled(True)
        self.generate_button.setEnabled(True)
    
    def display_media(self, media_path: str):
        """Display media
        
        Args:
            media_path: Path to media file
        """
        try:
            # Store current media path
            self.current_media_path = media_path
            
            # Check if path exists
            if not os.path.exists(media_path):
                self.media_display.setText(f"Media file not found: {media_path}")
                return
            
            # Check file extension
            is_video = media_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))
            
            if is_video:
                # For videos, show a placeholder and link to open
                self.media_display.setText(
                    f"Video generated at {media_path}\n\n"
                    "Click 'Save Media' to save or open it."
                )
            else:
                # For images, display directly
                pixmap = QPixmap(media_path)
                
                # Scale pixmap to fit label while preserving aspect ratio
                pixmap = pixmap.scaled(
                    self.media_display.width(), 
                    self.media_display.height(),
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.media_display.setPixmap(pixmap)
        except Exception as e:
            self.logger.error(f"Error displaying media: {str(e)}")
            self.media_display.setText(f"Error displaying media: {str(e)}")
    
    def save_media(self):
        """Save media to file"""
        if not self.current_media_path or not os.path.exists(self.current_media_path):
            return
        
        # Get file extension
        _, ext = os.path.splitext(self.current_media_path)
        
        # Determine file type filter based on extension
        if ext.lower() in ('.mp4', '.avi', '.mov', '.webm'):
            filter_str = "Video Files (*.mp4 *.avi *.mov *.webm);;All Files (*.*)"
        else:
            filter_str = "Image Files (*.png *.jpg *.jpeg *.gif);;All Files (*.*)"
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Media",
            "",
            filter_str
        )
        
        if not file_path:
            return
        
        try:
            # Copy file
            import shutil
            shutil.copy2(self.current_media_path, file_path)
            
            # Show success message in status bar if available
            main_window = self
            while main_window and not hasattr(main_window, 'status_bar'):
                main_window = main_window.parent()
            
            if main_window and hasattr(main_window, 'status_bar'):
                main_window.status_bar.showMessage(f"Media saved to {file_path}", 3000)
            else:
                # Fallback
                self.logger.info(f"Media saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving media: {str(e)}")
            # Try to show error message
            main_window = self
            while main_window and not hasattr(main_window, 'status_bar'):
                main_window = main_window.parent()
            
            if main_window and hasattr(main_window, 'status_bar'):
                main_window.status_bar.showMessage(f"Error saving media: {str(e)}", 3000)