"""
Style system for sagax1
Provides a professional and attractive UI style
"""

class StyleSystem:
    """Manages the styling for the sagax1 application"""
    
    # Professional color palette
    COLORS = {
        "primary": "#2D5F8B",       # Deep blue - primary brand color
        "primary_light": "#4F8BC9",  # Lighter blue for highlights
        "primary_dark": "#1D3F5B",   # Darker blue for accents
        "secondary": "#8C3D2F",     # Warm accent color (rust)
        "secondary_light": "#BF5E4C", # Lighter accent
        "accent": "#FCA311",        # Gold accent for important elements
        "success": "#4CAF50",       # Green for success states
        "warning": "#FF9800",       # Orange for warnings
        "error": "#F44336",         # Red for errors
        "light": "#F5F5F5",         # Light background
        "dark": "#333333",          # Dark text
        "gray": "#9E9E9E",          # Medium gray for subtle elements
        "light_gray": "#E0E0E0",    # Light gray for borders
        "white": "#FFFFFF",         # White
        "background": "#F9FBFD",    # Very light blue-gray background
    }
    
    # Main stylesheet with all components styled
    @staticmethod
    def get_main_stylesheet():
        """Get the main application stylesheet
        
        Returns:
            CSS stylesheet as string
        """
        colors = StyleSystem.COLORS
        
        return f"""
        /* Global styles */
        QMainWindow, QDialog {{
            background-color: {colors["background"]};
        }}
        
        /* Menu styling */
        QMenuBar {{
            background-color: {colors["primary"]};
            color: {colors["white"]};
            padding: 2px;
            spacing: 3px;
        }}
        
        QMenuBar::item {{
            background: transparent;
            padding: 4px 12px;
            border-radius: 4px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {colors["primary_light"]};
        }}
        
        QMenu {{
            background-color: {colors["white"]};
            border: 1px solid {colors["light_gray"]};
            border-radius: 4px;
            padding: 4px;
            color: black;
        }}
        
        QMenu::item {{
            padding: 6px 24px 6px 12px;
            border-radius: 3px;
            color: black;
        }}
        
        QMenu::item:selected {{
            background-color: {colors["primary_light"]};
            color: {colors["white"]};
        }}
        
        /* Tab widget styling */
        QTabWidget::pane {{
            border: 1px solid {colors["light_gray"]};
            border-radius: 4px;
            background-color: {colors["white"]};
            top: -1px;
        }}
        
        QTabBar::tab {{
            background-color: {colors["light"]};
            border: 1px solid {colors["light_gray"]};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 8px 16px;
            margin-right: 2px;
            color: {colors["dark"]};
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors["white"]};
            border-bottom-color: {colors["white"]};
            color: {colors["primary"]};
            font-weight: bold;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {colors["light_gray"]};
        }}
        
        /* Button styling */
        QPushButton {{
            background-color: {colors["primary"]};
            color: {colors["white"]};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            min-width: 80px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {colors["primary_light"]};
        }}
        
        QPushButton:pressed {{
            background-color: {colors["primary_dark"]};
        }}
        
        QPushButton:disabled {{
            background-color: {colors["light_gray"]};
            color: {colors["gray"]};
        }}
        
        /* Special action buttons */
        QPushButton[cssClass="action-button"] {{
            background-color: {colors["accent"]};
            font-weight: bold;
        }}
        
        QPushButton[cssClass="action-button"]:hover {{
            background-color: #FFB344;
        }}
        
        QPushButton[cssClass="action-button"]:pressed {{
            background-color: #E09400;
        }}
        
        /* Success button */
        QPushButton[cssClass="success-button"] {{
            background-color: {colors["success"]};
        }}
        
        QPushButton[cssClass="success-button"]:hover {{
            background-color: #5DBF61;
        }}
        
        /* Warning/Delete button */
        QPushButton[cssClass="warning-button"] {{
            background-color: {colors["warning"]};
        }}
        
        QPushButton[cssClass="warning-button"]:hover {{
            background-color: #FFA726;
        }}
        
        /* Danger/Error button */
        QPushButton[cssClass="danger-button"] {{
            background-color: {colors["error"]};
        }}
        
        QPushButton[cssClass="danger-button"]:hover {{
            background-color: #EF5350;
        }}
        
        /* Input styling */
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {{
            border: 1px solid {colors["light_gray"]};
            border-radius: 4px;
            padding: 8px;
            background-color: {colors["white"]};
            selection-background-color: {colors["primary_light"]};
            color: black;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
            border: 1px solid {colors["primary"]};
            color: black;
        }}
        
        QComboBox {{
            padding-right: 20px; /* Make space for the dropdown arrow */
        }}
        
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 20px;
            border-left: 1px solid {colors["light_gray"]};
        }}
        
        /* Labels */
        QLabel {{
            color: {colors["dark"]};
        }}
        
        QLabel[cssClass="heading"] {{
            font-size: 16pt;
            font-weight: bold;
            color: {colors["primary"]};
        }}
        
        QLabel[cssClass="subheading"] {{
            font-size: 12pt;
            font-weight: bold;
            color: {colors["primary_dark"]};
        }}
        
        /* List and tree widgets */
        QListWidget, QTreeWidget, QTableWidget {{
            border: 1px solid {colors["light_gray"]};
            border-radius: 4px;
            background-color: {colors["white"]};
            alternate-background-color: {colors["light"]};
            color: black; 
        }}
        
        QListWidget::item, QTreeWidget::item {{
            padding: 4px;
            border-radius: 2px;
            color: black; 
        }}
        
        QListWidget::item:selected, QTreeWidget::item:selected {{
            background-color: {colors["primary_light"]};
            color: {colors["white"]};
            color: black; 
        }}
        
        /* Status bar */
        QStatusBar {{
            background-color: {colors["primary"]};
            color: {colors["white"]};
            padding: 4px;
        }}
        
        /* Group box */
        QGroupBox {{
            border: 1px solid {colors["light_gray"]};
            border-radius: 4px;
            margin-top: 16px;
            padding-top: 16px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 8px;
            padding: 0 4px;
            color: {colors["primary"]};
            font-weight: bold;
        }}
        
        /* Scrollbar styling */
        QScrollBar:vertical {{
            border: none;
            background: {colors["light"]};
            width: 12px;
            margin: 12px 0 12px 0;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {colors["gray"]};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {colors["primary"]};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            border: none;
            background: {colors["light"]};
            height: 12px;
            margin: 0 12px 0 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {colors["gray"]};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background: {colors["primary"]};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            border: none;
            background: none;
            width: 0px;
        }}
        
        /* Progress bar */
        QProgressBar {{
            border: 1px solid {colors["light_gray"]};
            border-radius: 4px;
            text-align: center;
            background-color: {colors["white"]};
        }}
        
        QProgressBar::chunk {{
            background-color: {colors["primary"]};
            border-radius: 3px;
        }}
        
        /* CheckBox */
        QCheckBox {{
            spacing: 8px;
        }}
        
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {colors["light_gray"]};
            border-radius: 3px;
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {colors["primary"]};
            border: 1px solid {colors["primary"]};
            image: url(assets/icons/check.png);
        }}
        
        QCheckBox::indicator:unchecked:hover {{
            border: 1px solid {colors["primary"]};
        }}
        
        /* Radio Button */
        QRadioButton {{
            spacing: 8px;
        }}
        
        QRadioButton::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {colors["light_gray"]};
            border-radius: 9px;
        }}
        
        QRadioButton::indicator:checked {{
            background-color: {colors["primary"]};
            border: 1px solid {colors["primary"]};
            border-radius: 9px;
        }}
        
        QRadioButton::indicator:unchecked:hover {{
            border: 1px solid {colors["primary"]};
        }}
        """
    
    @staticmethod
    def apply_stylesheet(app):
        """Apply the stylesheet to the application
        
        Args:
            app: QApplication instance
        """
        app.setStyleSheet(StyleSystem.get_main_stylesheet())
    
    @staticmethod
    def create_action_button(button):
        """Style a button as a primary action button
        
        Args:
            button: QPushButton instance
        """
        button.setProperty("cssClass", "action-button")
        button.style().unpolish(button)
        button.style().polish(button)
    
    @staticmethod
    def create_success_button(button):
        """Style a button as a success button
        
        Args:
            button: QPushButton instance
        """
        button.setProperty("cssClass", "success-button")
        button.style().unpolish(button)
        button.style().polish(button)
    
    @staticmethod
    def create_warning_button(button):
        """Style a button as a warning button
        
        Args:
            button: QPushButton instance
        """
        button.setProperty("cssClass", "warning-button")
        button.style().unpolish(button)
        button.style().polish(button)
    
    @staticmethod
    def create_danger_button(button):
        """Style a button as a danger/error button
        
        Args:
            button: QPushButton instance
        """
        button.setProperty("cssClass", "danger-button")
        button.style().unpolish(button)
        button.style().polish(button)
    
    @staticmethod
    def style_heading(label):
        """Style a label as a heading
        
        Args:
            label: QLabel instance
        """
        label.setProperty("cssClass", "heading")
        label.style().unpolish(label)
        label.style().polish(label)
    
    @staticmethod
    def style_subheading(label):
        """Style a label as a subheading
        
        Args:
            label: QLabel instance
        """
        label.setProperty("cssClass", "subheading")
        label.style().unpolish(label)
        label.style().polish(label)