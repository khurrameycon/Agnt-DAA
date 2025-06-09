"""
UI Assets Manager for sagax1
Handles loading of icons and other UI assets
"""

import os
from PyQt6.QtGui import QIcon, QPixmap, QFont, QFontDatabase
from PyQt6.QtCore import QSize, Qt, QBuffer, QByteArray, QIODevice

class UIAssets:
    """Manages UI assets like icons and fonts for the application"""
    
    # Icon mapping
    ICONS = {
        "app": "sagax1-logo.png",
        "send": "send.png",
        "refresh": "refresh.png",
        "create": "add.png",
        "delete": "delete.png",
        "settings": "settings.png",
        "search": "search.png",
        "code": "code.png",
        "web": "web.png",
        "visual": "visual.png",
        "media": "media.png",
        "fine_tuning": "tune.png",
        "chat": "chat.png",
        "agents": "agents.png",
        "back": "back.png",
        "forward": "forward.png",
        "save": "save.png",
        "clear": "clear.png",
        "copy": "copy.png",
        "cancel": "cancel.png",
        "ok": "check.png",
        "info": "info.png",
        "warning": "warning.png",
        "error": "error.png",
        "success": "success.png"
    }
    
    # Default icons in case the file doesn't exist
    DEFAULT_ICONS = {
        "app": """
            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
                <rect width="64" height="64" fill="#2D5F8B" rx="12" ry="12"/>
                <text x="32" y="38" font-family="Arial" font-size="24" fill="white" text-anchor="middle">S1</text>
            </svg>
        """,
        "send": """
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <path fill="#2D5F8B" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
        """,
        "refresh": """
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <path fill="#2D5F8B" d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
            </svg>
        """,
        "add": """
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <path fill="#2D5F8B" d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
            </svg>
        """,
        "delete": """
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <path fill="#F44336" d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
            </svg>
        """,
        "settings": """
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <path fill="#2D5F8B" d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
            </svg>
        """,
        "search": """
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <path fill="#2D5F8B" d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
            </svg>
        """,
        "check": """
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                <path fill="#4CAF50" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>
        """
    }
    
    # Folder with icons
    ICON_FOLDER = os.path.join("assets", "icons")
    
    @staticmethod
    def create_default_icons_file():
        """
        Create default icons when QApplication is available
        This should be called after QApplication is initialized
        """
        try:
            # Create minimal set of necessary icons if they don't exist
            for name, svg in UIAssets.DEFAULT_ICONS.items():
                icon_path = os.path.join(UIAssets.ICON_FOLDER, UIAssets.ICONS[name])
                
                if not os.path.exists(icon_path):
                    try:
                        # Create folder if it doesn't exist
                        os.makedirs(os.path.dirname(icon_path), exist_ok=True)
                        
                        # Convert SVG to PNG and save
                        svg_data = svg.encode('utf-8')
                        pixmap = QPixmap()
                        pixmap.loadFromData(svg_data)
                        
                        # Save as PNG
                        pixmap.save(icon_path, "PNG")
                    except Exception as e:
                        print(f"Error creating icon {name}: {str(e)}")
        except Exception as e:
            print(f"Error creating default icons: {str(e)}")
    
    @staticmethod
    def ensure_assets_exist():
        """Ensure that the assets folder exists"""
        # Create icons folder if it doesn't exist
        os.makedirs(UIAssets.ICON_FOLDER, exist_ok=True)
        
        # NOTE: We don't create the default icons here anymore
        # They will be created in create_default_icons_file() after QApplication exists
    
    @staticmethod
    def get_icon(name, size=None):
        """Get an icon by name
        
        Args:
            name: Icon name
            size: Optional size tuple (width, height)
            
        Returns:
            QIcon instance
        """
        # Ensure icon exists in mapping
        if name not in UIAssets.ICONS:
            return QIcon()
        
        icon_path = os.path.join(UIAssets.ICON_FOLDER, UIAssets.ICONS[name])
        
        # Check if icon file exists
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            # Use default SVG icon if file doesn't exist
            if name in UIAssets.DEFAULT_ICONS:
                svg_data = UIAssets.DEFAULT_ICONS[name].encode('utf-8')
                pixmap = QPixmap()
                pixmap.loadFromData(svg_data)
                icon = QIcon(pixmap)
            else:
                icon = QIcon()
        
        # Set specific size if requested
        if size and icon and not icon.isNull():
            pixmap = icon.pixmap(QSize(*size))
            icon = QIcon(pixmap)
        
        return icon
    
    @staticmethod
    def get_pixmap(name, width=None, height=None):
        """Get a pixmap by name
        
        Args:
            name: Icon name
            width: Optional width
            height: Optional height
            
        Returns:
            QPixmap instance
        """
        # Ensure icon exists in mapping
        if name not in UIAssets.ICONS:
            return QPixmap()
        
        icon_path = os.path.join(UIAssets.ICON_FOLDER, UIAssets.ICONS[name])
        
        # Check if icon file exists
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
        else:
            # Use default SVG icon if file doesn't exist
            if name in UIAssets.DEFAULT_ICONS:
                svg_data = UIAssets.DEFAULT_ICONS[name].encode('utf-8')
                pixmap = QPixmap()
                pixmap.loadFromData(svg_data)
            else:
                pixmap = QPixmap()
        
        # Resize if dimensions are specified
        if width and height and not pixmap.isNull():
            pixmap = pixmap.scaled(
                width, height, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
        elif width and not pixmap.isNull():
            pixmap = pixmap.scaledToWidth(
                width, Qt.TransformationMode.SmoothTransformation
            )
        elif height and not pixmap.isNull():
            pixmap = pixmap.scaledToHeight(
                height, Qt.TransformationMode.SmoothTransformation
            )
        
        return pixmap
    
    @staticmethod
    def apply_app_icon(app):
        """Apply the application icon
        
        Args:
            app: QApplication instance
        """
        app.setWindowIcon(UIAssets.get_icon("app"))