"""
Sample demo tasks for the Visual Web Agent
This can be used to showcase the capabilities of the Visual Web Agent
"""

class VisualWebDemo:
    """Sample demo tasks for the Visual Web Agent"""
    
    @staticmethod
    def get_demo_tasks():
        """Get a list of sample demo tasks
        
        Returns:
            List of demo task dictionaries
        """
        return [
            {
                "name": "Wikipedia Search",
                "description": "Navigate to Wikipedia and search for a topic",
                "prompt": "Go to wikipedia.org, search for 'Artificial Intelligence', and tell me the first sentence of the article."
            },
            {
                "name": "GitHub Trending",
                "description": "Check trending repositories on GitHub",
                "prompt": "Visit github.com/trending and tell me the top 3 trending repositories today."
            },
            {
                "name": "Weather Lookup",
                "description": "Check the weather forecast for a city",
                "prompt": "Go to weather.com and check the weather forecast for New York City for the next 3 days."
            },
            {
                "name": "News Summary",
                "description": "Get headlines from a news website",
                "prompt": "Navigate to news.ycombinator.com and summarize the top 5 stories."
            },
            {
                "name": "Product Search",
                "description": "Search for products on Amazon",
                "prompt": "Go to amazon.com, search for 'bluetooth headphones', and tell me the name and price of the top 3 results."
            }
        ]

    @staticmethod
    def add_demo_menu(visual_web_tab):
        """Add a demo menu to the visual web tab
        
        Args:
            visual_web_tab: Visual web tab instance
        """
        from PyQt6.QtWidgets import QPushButton, QMenu
        from PyQt6.QtGui import QAction  # Import QAction from QtGui instead of QtWidgets
        
        # Create demo button
        demo_button = QPushButton("Run Demo")
        visual_web_tab.layout.itemAt(0).layout().addWidget(demo_button)
        
        # Create menu
        demo_menu = QMenu()
        
        # Add demo tasks
        for task in VisualWebDemo.get_demo_tasks():
            action = QAction(task["name"], demo_menu)
            action.setStatusTip(task["description"])
            action.triggered.connect(lambda checked, t=task: VisualWebDemo.run_demo_task(visual_web_tab, t))
            demo_menu.addAction(action)
        
        # Set menu on button
        demo_button.setMenu(demo_menu)
    
    @staticmethod
    def run_demo_task(visual_web_tab, task):
        """Run a demo task
        
        Args:
            visual_web_tab: Visual web tab instance
            task: Task dictionary
        """
        # Make sure browser is started
        if not visual_web_tab.current_agent or not hasattr(visual_web_tab.current_agent, 'visual_tool') or visual_web_tab.current_agent.visual_tool.browser is None:
            # Try to start browser
            visual_web_tab.toggle_browser()
            
            # Check if browser started
            if not visual_web_tab.current_agent or not hasattr(visual_web_tab.current_agent, 'visual_tool') or visual_web_tab.current_agent.visual_tool.browser is None:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    visual_web_tab,
                    "Browser Not Started",
                    "Please start the browser first by clicking the 'Start Browser' button."
                )
                return
        
        # Set the prompt in the input field
        visual_web_tab.command_input.setText(task["prompt"])
        
        # Send the command
        visual_web_tab.send_command()