"""
Simplified Visual Web Tab Component for sagax1
Tab for installing and launching the web-ui application
"""

import os
import logging
import sys
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QSplitter, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QProcess, QUrl, pyqtSlot
from PyQt6.QtGui import QFont, QTextCursor
import uuid
import re
from PyQt6.QtCore import Qt, QProcess, QUrl, pyqtSlot, QTimer
# Correct imports for PyQt6-WebEngine
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    # Provide a helpful error message if the module is not found
    from PyQt6.QtWidgets import QLabel
    class QWebEngineView(QLabel):
        """Placeholder for QWebEngineView when PyQt6-WebEngine is not installed"""
        def __init__(self, *args, **kwargs):
            super().__init__("WebEngineView not available. Please install PyQt6-WebEngine package.", *args, **kwargs)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setWordWrap(True)
            self.setStyleSheet("background-color: #f0f0f0; padding: 20px; border: 1px solid #ccc;")
        
        def setUrl(self, url):
            self.setText(f"WebEngineView not available. Cannot navigate to {url.toString()}\nPlease install PyQt6-WebEngine package.")

from app.core.agent_manager import AgentManager


class VisualWebTab(QWidget):
    """Simplified tab for installing and launching web-ui"""
    
    def __init__(self, agent_manager: AgentManager, parent=None):
        """Initialize the visual web tab
        
        Args:
            agent_manager: Agent manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.agent_manager = agent_manager
        self.logger = logging.getLogger(__name__)
        
        # Process instances
        self.install_process = None
        self.launch_process = None
        self.is_installed = False
        self.is_launched = False
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create buttons
        self.create_button_panel()
        
        # Create main panels
        self.create_main_panels()
        
        # Add status bar
        self.create_status_bar()
        
        # Check if files exist
        self.check_files_exist()
    
    def create_button_panel(self):
        """Create panel with installation and launch buttons"""
        button_layout = QHBoxLayout()
        
        # Install button
        self.install_button = QPushButton("Install/Initialize Visual Agent")
        self.install_button.setToolTip("Run new.bat to set up environment and install web-ui")
        self.install_button.clicked.connect(self.install_visual_agent)
        button_layout.addWidget(self.install_button)
        
        # Launch button
        self.launch_button = QPushButton("Launch Visual Agent")
        self.launch_button.setToolTip("Run testing-web-ui.py to start the web-ui server")
        self.launch_button.clicked.connect(self.launch_visual_agent)
        # self.launch_button.setEnabled(False)  # Disabled until installation completes
        button_layout.addWidget(self.launch_button)
        
        # Add space
        button_layout.addStretch()
        
        # Add to main layout
        self.layout.addLayout(button_layout)
    
    def create_main_panels(self):
        """Create main panels for browser view and output"""
        # Create splitter
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Create browser panel (top)
        self.browser_panel = QWidget()
        browser_layout = QVBoxLayout(self.browser_panel)
        
        # Create web view
        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl("about:blank"))  # Start with blank page
        browser_layout.addWidget(self.web_view)
        
        # Create browser status
        self.browser_status = QLabel("Browser not yet loaded. Please install and launch the Visual Agent first.")
        self.browser_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.browser_status.setStyleSheet("color: #666; font-style: italic;")
        browser_layout.addWidget(self.browser_status)
        
        # Create output panel (bottom)
        self.output_panel = QWidget()
        output_layout = QVBoxLayout(self.output_panel)
        
        # Create output display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setFont(QFont("Courier New", 9))  # Monospace font for output
        self.output_display.setPlaceholderText("Output will be displayed here...")
        output_layout.addWidget(self.output_display)
        
        # Add panels to splitter
        self.splitter.addWidget(self.browser_panel)
        self.splitter.addWidget(self.output_panel)
        
        # Set initial sizes (more space for browser)
        self.splitter.setSizes([600, 300])
        
        # Add to main layout
        self.layout.addWidget(self.splitter, 1)  # stretch=1 to take all available space
    
    def create_status_bar(self):
        """Create status bar with progress indicator"""
        status_layout = QHBoxLayout()
        
        # Status label
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label, 1)  # stretch=1 to take available space
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        status_layout.addWidget(self.progress_bar)
        
        # Add to main layout
        self.layout.addLayout(status_layout)
    
    def check_files_exist(self):
        """Check if the required script files exist"""
        # Path to new.bat and testing-web-ui.py
        # Assume they are in the current directory for simplicity
        # You may need to adjust paths based on your setup
        self.new_bat_path = r"app\utils\new.bat"
        self.testing_web_ui_path = r"app\utils\testing-web-ui.py"
        
        files_exist = True
        missing_files = []
        
        if not os.path.exists(self.new_bat_path):
            files_exist = False
            missing_files.append(self.new_bat_path)
        
        if not os.path.exists(self.testing_web_ui_path):
            files_exist = False
            missing_files.append(self.testing_web_ui_path)
        
        if not files_exist:
            self.log_output("Warning: Could not find the following required files:")
            for file in missing_files:
                self.log_output(f"  - {file}")
            self.log_output("Please make sure these files are in the correct location.")
            
            self.install_button.setEnabled(False)
            # Keep launch button enabled regardless
            self.update_status("Missing required files", is_error=True)
        else:
            # Both buttons should be enabled
            self.install_button.setEnabled(True)
            self.update_status("Ready")
        
        # Always keep the launch button enabled
        self.launch_button.setEnabled(True)
        # In the check_files_exist method, add this line:
        self.install_button.setEnabled(not os.path.exists(os.path.join(os.path.expanduser("~"), ".sagax1", "web-ui")))

    
    def install_visual_agent(self):
        """Run the new.bat script with administrator privileges to install the Visual Agent"""
        self.log_output("\n--- Starting Installation Process ---")
        self.log_output(f"Running: {self.new_bat_path} (with administrator privileges)")
        
        # Disable buttons during installation
        self.install_button.setEnabled(False)
        # self.launch_button.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.update_status("Installing Visual Agent...")
        
        try:
            if sys.platform == 'win32':
                # Using only ctypes for Windows UAC elevation
                import ctypes
                import tempfile
                import uuid
                
                # Create a temporary batch file that will capture output
                temp_dir = tempfile.gettempdir()
                output_file = os.path.join(temp_dir, f"visual_agent_install_output_{uuid.uuid4().hex}.txt")
                self.log_output(f"Output will be captured to: {output_file}")
                
                # Create a wrapper batch file to run the actual script with output redirection
                wrapper_batch = os.path.join(temp_dir, f"run_as_admin_{uuid.uuid4().hex}.bat")
                with open(wrapper_batch, 'w') as f:
                    # Get absolute path to new.bat
                    abs_path = os.path.abspath(self.new_bat_path)
                    # Write batch file to run new.bat and redirect output
                    f.write(f'@echo off\n')
                    f.write(f'echo Running {abs_path} with Administrator privileges...\n')
                    f.write(f'cd /d "{os.path.dirname(abs_path)}"\n')  # Change to the correct directory
                    f.write(f'call "{abs_path}" > "{output_file}" 2>&1\n')
                    f.write(f'echo Installation completed with exit code %ERRORLEVEL% >> "{output_file}"\n')
                    f.write(f'exit %ERRORLEVEL%\n')
                
                # Run the wrapper batch file with admin privileges
                self.log_output("Requesting administrator privileges...")
                result = ctypes.windll.shell32.ShellExecuteW(
                    None,                   # Parent window handle
                    "runas",                # Operation - "runas" means run as admin
                    "cmd.exe",              # Application to run
                    f'/c "{wrapper_batch}"',# Parameters - run our wrapper batch
                    None,                   # Directory - None means current directory
                    1                       # Window state - normal (SW_SHOWNORMAL)
                )
                
                # Check if UAC prompt was shown successfully (result > 32 means success)
                if result <= 32:
                    error_msg = f"Failed to request administrator privileges. Error code: {result}"
                    self.log_output(error_msg)
                    self.install_button.setEnabled(True)
                    self.progress_bar.setVisible(False)
                    self.update_status("Installation failed", is_error=True)
                    return
                
                # Set up a timer to check for the output file
                self.output_file = output_file
                self.wrapper_file = wrapper_batch
                self.output_timer = QTimer(self)
                self.output_timer.timeout.connect(self.check_install_output)
                self.output_timer.start(1000)  # Check every second
                
                self.log_output("Administrator privileges granted. Installation running...")
                
            else:
                # For non-Windows platforms, show a message that admin rights are needed
                self.log_output("This installation requires administrator privileges.")
                self.log_output("Please run the batch file manually with sudo or equivalent.")
                self.install_button.setEnabled(True)
                self.progress_bar.setVisible(False)
                self.update_status("Manual installation required", is_error=True)
                
        except Exception as e:
            self.log_output(f"Error starting installation: {str(e)}")
            import traceback
            self.log_output(traceback.format_exc())
            self.install_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.update_status("Installation failed", is_error=True)

    def check_install_output(self):
        """Check for output from the elevated installation process"""
        if hasattr(self, 'output_file') and os.path.exists(self.output_file):
            try:
                # Read the current content of the output file
                with open(self.output_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    
                # Check if we've already displayed this content
                if not hasattr(self, 'last_output_position'):
                    self.last_output_position = 0
                    
                # Only display new content
                if len(content) > self.last_output_position:
                    new_content = content[self.last_output_position:]
                    self.log_output(new_content, end='')
                    self.last_output_position = len(content)
                    
                # Check if installation is complete or successful indicators in the output
                if "Installation completed with exit code" in content or "Installation completed successfully" in content:
                    # Extract exit code if available
                    exit_code_match = re.search(r"Installation completed with exit code (\d+)", content)
                    exit_code = 0
                    if exit_code_match:
                        exit_code = int(exit_code_match.group(1))
                    
                    # Stop the timer
                    self.output_timer.stop()
                    
                    # Clean up temporary files
                    if hasattr(self, 'wrapper_file') and os.path.exists(self.wrapper_file):
                        try:
                            os.remove(self.wrapper_file)
                        except:
                            pass
                    
                    # Consider installation successful
                    self.log_output("\nInstallation completed!")
                    self.update_status("Installation completed")
                    self.is_installed = True
                    
                    # Always enable launch button when installation completes
                    self.launch_button.setEnabled(True)
                    self.progress_bar.setVisible(False)
                    
            except Exception as e:
                self.log_output(f"Error reading install output: {str(e)}")
                
        # If the file doesn't exist after a while, assume something went wrong
        elif hasattr(self, 'output_check_count'):
            self.output_check_count += 1
            if self.output_check_count > 30:  # 30 seconds timeout
                self.output_timer.stop()
                self.log_output("Timeout waiting for installation output. The installation may have failed.")
                self.install_button.setEnabled(True)
                self.progress_bar.setVisible(False)
                self.update_status("Installation timed out", is_error=True)
        else:
            self.output_check_count = 0
    
    def launch_visual_agent(self):
        """Run the web-ui launch code directly"""
        self.log_output("\n--- Starting Launch Process ---")
        
        # Define the configuration constants
        USER_HOME = os.path.expanduser("~")  # Gets user's home directory
        WEBUI_DIRECTORY = os.path.join(USER_HOME, ".sagax1", "web-ui")
        
        # Log the configuration
        self.log_output(f"Attempting to launch WebUI from: {WEBUI_DIRECTORY}")
        
        # Construct the full paths
        venv_python_executable = os.path.join(WEBUI_DIRECTORY, '.venv', 'Scripts', 'python.exe')
        webui_script = os.path.join(WEBUI_DIRECTORY, 'webui.py')
        
        self.log_output(f"Using Python executable: {venv_python_executable}")
        self.log_output(f"Running script: {webui_script}")
        
        # Check if the virtual environment Python and the script exist
        if not os.path.exists(venv_python_executable):
            self.log_output(f"\nError: Python executable not found at '{venv_python_executable}'")
            self.log_output("Please ensure the virtual environment exists and the path is correct.")
            self.update_status("Launch failed - missing Python executable", is_error=True)
            return
        
        if not os.path.exists(webui_script):
            self.log_output(f"\nError: WebUI script not found at '{webui_script}'")
            self.log_output("Please ensure the web-ui installation is correct.")
            self.update_status("Launch failed - missing WebUI script", is_error=True)
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.update_status("Launching Visual Agent...")
        
        try:
            self.log_output("\nLaunching WebUI...")
            
            # For Windows, use subprocess directly to create a new console window
            if sys.platform == 'win32':
                import subprocess
                
                # Create the process in a new console window
                process = subprocess.Popen(
                    [venv_python_executable, webui_script],
                    cwd=WEBUI_DIRECTORY,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                self.log_output("Launch command sent. The WebUI should start in a new window.")
                self.log_output(f"Process ID: {process.pid}")
                
                # Load the web UI after a short delay
                QTimer.singleShot(9000, self.load_web_ui)
                
                # Update status
                self.update_status("Web UI launched")
                self.is_launched = True
                self.progress_bar.setVisible(False)

            else:
                # For non-Windows, use QProcess
                self.launch_process = QProcess()
                self.launch_process.setWorkingDirectory(WEBUI_DIRECTORY)
                self.launch_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
                
                # Connect signals
                self.launch_process.readyRead.connect(self.on_launch_output)
                self.launch_process.finished.connect(self.on_launch_finished)
                
                # Start the process
                self.launch_process.start(venv_python_executable, [webui_script])
                
        except Exception as e:
            self.log_output(f"\nAn error occurred while trying to launch the WebUI: {e}")
            import traceback
            self.log_output(traceback.format_exc())
            self.progress_bar.setVisible(False)
            self.update_status("Launch failed", is_error=True)
    
    @pyqtSlot()
    def on_install_output(self):
        """Handle output from the installation process"""
        if self.install_process:
            output = self.install_process.readAll().data().decode('utf-8', errors='replace')
            self.log_output(output, end='')
    
    @pyqtSlot(int, QProcess.ExitStatus)
    def on_install_finished(self, exit_code, exit_status):
        """Handle completion of the installation process"""
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self.log_output("\nInstallation completed successfully!")
            self.update_status("Installation completed")
            self.is_installed = True
            
            # Enable launch button
            self.launch_button.setEnabled(True)
        else:
            self.log_output(f"\nInstallation process exited with code {exit_code}")
            self.update_status("Installation failed", is_error=True)
            
            # Re-enable install button for retry
            self.install_button.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
    
    @pyqtSlot()
    def on_launch_output(self):
        """Handle output from the launch process"""
        if self.launch_process:
            output = self.launch_process.readAll().data().decode('utf-8', errors='replace')
            self.log_output(output, end='')
            
            # Check for successful launch indicators in the output
            if "WebUI should start in a new window" in output:
                # Server successfully started, load the web page
                self.load_web_ui()
    
    @pyqtSlot(int, QProcess.ExitStatus)
    def on_launch_finished(self, exit_code, exit_status):
        """Handle completion of the launch process"""
        # Note: For a successful launch, the process stays running
        # If we get here, it means the process exited unexpectedly
        
        self.log_output(f"\nLaunch process exited with code {exit_code}")
        
        if self.is_launched:
            self.update_status("Web UI is running")
        else:
            self.update_status("Launch failed", is_error=True)
            # Re-enable launch button for retry
            self.launch_button.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
    
    def load_web_ui(self):
        """Load the Web UI in the browser view"""
        self.log_output("Loading Web UI in browser view...")
        self.is_launched = True
        
        # Set the URL to the web UI
        self.web_view.setUrl(QUrl("http://127.0.0.1:7788"))
        
        # Update browser status
        self.browser_status.setText("Web UI loaded. If the page doesn't appear, click the refresh button.")
        
        # Update status
        self.update_status("Web UI running")
        self.progress_bar.setVisible(False)
    
    def log_output(self, text, end='\n'):
        """Add text to the output display
        
        Args:
            text: Text to add
            end: End string (default: newline)
        """
        self.output_display.moveCursor(QTextCursor.MoveOperation.End)
        self.output_display.insertPlainText(text + end)
        self.output_display.moveCursor(QTextCursor.MoveOperation.End)
        
        # Also log to application log
        self.logger.info(text.rstrip())
    
    def update_status(self, status: str, is_error: bool = False):
        """Update the status label
        
        Args:
            status: Status text
            is_error: Whether this is an error status
        """
        self.status_label.setText(status)
        
        if is_error:
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: black;")
    
    def closeEvent(self, event):
        """Handle the tab being closed
        
        Args:
            event: Close event
        """
        # Terminate any running processes
        if self.install_process and self.install_process.state() == QProcess.ProcessState.Running:
            self.install_process.terminate()
            self.install_process.waitForFinished(1000)  # Wait up to 1 second
        
        if self.launch_process and self.launch_process.state() == QProcess.ProcessState.Running:
            # Note: We don't terminate the launch process by default as it runs the web UI
            # Uncomment the following to terminate on tab close if desired
            # self.launch_process.terminate()
            # self.launch_process.waitForFinished(1000)
            pass
        
        super().closeEvent(event)