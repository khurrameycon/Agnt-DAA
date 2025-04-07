# sagax1

## An Opensource AI-powered agent platform for everyday tasks

sagax1 is a powerful application that allows you to create and use various AI agents to perform tasks like:
- Browsing the web
- Generating images
- Writing code
- Visual web automation
- Fine-tuning models

## Installation

### Windows Installation

1. Download the latest installer (`sagax1_Setup.exe`) from the releases page
2. Run the installer and follow the instructions
3. Launch sagax1 from the desktop shortcut or start menu

### Developer Installation

To set up the development environment:

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sagax1.git
   cd sagax1
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python main.py
   ```

### Building the Installer

To build the Windows installer:

1. Make sure you have Python 3.8+ and NSIS installed
2. Run the build script:
   ```
   build_installer.bat
   ```
3. The installer will be created in the `dist` folder

## Usage

1. Launch sagax1
2. Create a new agent using the "Create New" button
3. Select the agent type based on your task
4. Configure the agent with your preferred settings
5. Start using the agent for your tasks

## Requirements

- Windows 10/11 (64-bit)
- Internet connection
- For visual web automation: Google Chrome must be installed

## API Keys

To use the full functionality of sagax1, you'll need to set up the following API keys:
- Hugging Face API key (for accessing models)

You can input these keys in the Settings tab of the application.

## License

Copyright © 2025 sagax1 Team