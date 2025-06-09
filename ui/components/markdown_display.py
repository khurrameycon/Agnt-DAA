"""
Integration Guide: Connecting WebBrowsingAgent with Markdown UI

This guide shows how to connect the WebBrowsingAgent with the MarkdownResultsDisplay
component to properly display markdown-formatted search results.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, pyqtSignal

# Import your WebBrowsingAgent
from app.agents.web_browsing_agent import WebBrowsingAgent

# Import the MarkdownResultsDisplay
from .markdown_display import MarkdownResultsDisplay  # Assuming you saved the file as markdown_display.py

class WebBrowserTab(QWidget):
    """Web browser tab for sagax1 app
    
    This tab provides a UI for the WebBrowsingAgent, displaying web search
    results in a proper markdown format.
    """
    
    # Signal for when the agent is busy
    agent_busy = pyqtSignal(bool)
    
    def __init__(self, agent_manager, config_manager, parent=None):
        """Initialize the web browser tab
        
        Args:
            agent_manager: Agent manager instance
            config_manager: Config manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.agent_manager = agent_manager
        self.config_manager = config_manager
        self.agent = None
        
        # Initialize UI
        self.init_ui()
        
        # Initialize agent
        self._initialize_agent()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Search input area
        search_layout = QHBoxLayout()
        
        self.search_label = QLabel("Web Search:")
        search_layout.addWidget(self.search_label)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter your search query here...")
        self.search_input.returnPressed.connect(self.on_search)
        search_layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.on_search)
        search_layout.addWidget(self.search_button)
        
        main_layout.addLayout(search_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Markdown display for results
        self.results_display = MarkdownResultsDisplay(self)
        main_layout.addWidget(self.results_display)
        
        # Set layout
        self.setLayout(main_layout)
    
    def _initialize_agent(self):
        """Initialize the WebBrowsingAgent"""
        # Get default config from agent_manager or use default values
        agent_config = {
            "model_id": self.config_manager.get_config().get("models", {}).get("default_model", "meta-llama/Llama-3.2-3B-Instruct"),
            "use_api": True,  # Default to API mode for better performance
            "max_tokens": 2048,
            "temperature": 0.7
        }
        
        # Create the agent
        self.agent = WebBrowsingAgent("web_search", agent_config)
        
        # Initialize the agent
        self.agent.initialize()
    
    def on_search(self):
        """Handle search button click or Enter key press"""
        query = self.search_input.text().strip()
        if not query:
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.search_button.setEnabled(False)
        self.search_input.setEnabled(False)
        self.agent_busy.emit(True)
        
        # Function to update progress
        def update_progress(message):
            current_value = self.progress_bar.value()
            if "Searching" in message:
                self.progress_bar.setValue(30)
            elif "Processing" in message:
                self.progress_bar.setValue(60)
            else:
                # Increment by 5% for any other message
                self.progress_bar.setValue(min(current_value + 5, 95))
        
        # Run the search in a separate thread to keep UI responsive
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class SearchThread(QThread):
            search_completed = pyqtSignal(str, str)
            progress_update = pyqtSignal(str)
            
            def __init__(self, agent, query):
                super().__init__()
                self.agent = agent
                self.query = query
            
            def run(self):
                try:
                    # Pass the progress update callback
                    result = self.agent.run(self.query, callback=lambda msg: self.progress_update.emit(msg))
                    self.search_completed.emit(result, self.query)
                except Exception as e:
                    self.search_completed.emit(f"Error: {str(e)}", self.query)
        
        # Create and start the thread
        self.search_thread = SearchThread(self.agent, query)
        self.search_thread.progress_update.connect(update_progress)
        self.search_thread.search_completed.connect(self.on_search_completed)
        self.search_thread.start()
    
    def on_search_completed(self, result, query):
        """Handle search completion
        
        Args:
            result: Search result text
            query: Search query
        """
        # Update UI
        self.progress_bar.setValue(100)
        self.search_button.setEnabled(True)
        self.search_input.setEnabled(True)
        self.agent_busy.emit(False)
        
        # Hide progress bar after a delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
        
        # Display the results in markdown format
        self.results_display.display_markdown(result, query)
    
    def clear_results(self):
        """Clear the results display"""
        self.results_display.clear()