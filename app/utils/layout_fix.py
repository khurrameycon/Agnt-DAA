"""
Fix for layout issues in the MainWindow to ensure content expands to fill the entire window
"""

from PyQt6.QtWidgets import QSizePolicy, QSplitter
from PyQt6.QtCore import Qt

def fix_main_window_layout(window):
    """Apply layout fixes to make the main window content expand to fill the entire window
    
    Args:
        window: The MainWindow instance
    """
    try:
        # Ensure the central widget expands in both directions
        central_widget = window.centralWidget()
        central_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Make sure the tabs widget fills the available space
        window.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # For each tab, ensure it expands
        for tab_index in range(window.tabs.count()):
            tab_widget = window.tabs.widget(tab_index)
            if tab_widget:
                tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                
                # For splitters inside tabs
                splitters = tab_widget.findChildren(QSplitter)
                for splitter in splitters:
                    # Make the splitter expand in both directions
                    splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    
                    # Ensure the splitter's children have reasonable sizes
                    # Use a try/except block because this can fail if the splitter is not yet visible
                    try:
                        if splitter.count() > 0:
                            sizes = [splitter.width() // splitter.count() for _ in range(splitter.count())]
                            splitter.setSizes(sizes)
                            
                            # Set stretch factors for all the children
                            for i in range(splitter.count()):
                                splitter.setStretchFactor(i, 1)
                    except Exception as e:
                        print(f"Error adjusting splitter sizes: {str(e)}")
        
    except Exception as e:
        print(f"Error in fix_main_window_layout: {str(e)}")