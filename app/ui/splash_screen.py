"""
Splash screen for SagaX1
"""

import os
from PyQt6.QtWidgets import QSplashScreen, QApplication
from PyQt6.QtGui import QPixmap, QColor, QPainter, QFont
from PyQt6.QtCore import Qt, QTimer, QSize

class SagaX1SplashScreen(QSplashScreen):
    """Custom splash screen for SagaX1 with professional appearance"""
    
    def __init__(self):
        """Initialize the splash screen with logo and styling"""
        # Create pixmap for the splash screen
        pixmap = self._create_splash_pixmap()
        
        # Initialize with the created pixmap
        super().__init__(pixmap)
        
        # Set window flags for modern appearance
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # Center on screen
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def _create_splash_pixmap(self):
        """Create the pixmap for the splash screen
        
        Returns:
            QPixmap instance
        """
        # Check if logo exists, otherwise create a text-based splash
        logo_path = os.path.join("assets", "icons", "sagax1-logo.png")
        
        if os.path.exists(logo_path):
            # Use the logo for the splash screen
            pixmap = QPixmap(logo_path)
            
            # Ensure the splash screen is reasonably sized
            if pixmap.width() > 600 or pixmap.height() > 400:
                pixmap = pixmap.scaled(QSize(600, 400), Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        else:
            # Create a stylized text splash screen as fallback
            pixmap = QPixmap(500, 300)
            pixmap.fill(QColor(30, 30, 60))  # Dark blue background
            
            # Create painter for custom drawing
            painter = QPainter(pixmap)
            
            # Set up font for title
            title_font = QFont("Arial", 36, QFont.Weight.Bold)
            painter.setFont(title_font)
            painter.setPen(QColor(255, 255, 255))  # White text
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "SagaX1")
            
            # Set up font for tagline
            tagline_font = QFont("Arial", 14)
            painter.setFont(tagline_font)
            painter.setPen(QColor(180, 180, 220))  # Light blue text
            
            # Position the tagline below the title
            tagline_rect = pixmap.rect()
            tagline_rect.setTop(tagline_rect.top() + 180)
            painter.drawText(tagline_rect, Qt.AlignmentFlag.AlignHCenter, 
                           "AI-powered agent platform for everyday tasks")
            
            # Finish painting
            painter.end()
            
        return pixmap
    
    def drawContents(self, painter):
        """Override to customize the appearance during loading"""
        super().drawContents(painter)
    
    def show_with_timer(self, app, main_window, duration=2000):
        """Show splash screen for specified duration then show main window
        
        Args:
            app: QApplication instance
            main_window: Main window to show after splash
            duration: Time in milliseconds to show splash
        """
        # Show the splash screen
        self.show()
        
        # Process events to ensure splash is displayed
        app.processEvents()
        
        # Set timer to close splash and show main window
        QTimer.singleShot(duration, lambda: self.finish_splash(main_window))
    
    def finish_splash(self, main_window):
        """Finish splash screen with fade effect and show main window
        
        Args:
            main_window: Main window to show after splash
        """
        # Hide splash and show main window
        self.finish(main_window)
        main_window.show()