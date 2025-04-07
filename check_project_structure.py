# Save as check_project_structure.py
import os
import sys

def check_structure():
    """Check if the project structure is valid for building"""
    print("Checking project structure...")
    
    # Required files
    required_files = [
        "main.py", 
        "requirements.txt", 
        "main.spec", 
        "installer.nsi",
        "create_icon.py",
        "create_installer_graphics.py"
    ]
    
    # Required directories
    required_dirs = [
        "app", 
        "app/core", 
        "app/agents", 
        "app/ui", 
        "app/utils", 
        "assets", 
        "config"
    ]
    
    # Check required files
    missing_files = []
    for file in required_files:
        if not os.path.isfile(file):
            missing_files.append(file)
    
    # Check required directories
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.isdir(directory):
            missing_dirs.append(directory)
    
    # Report issues
    if missing_files or missing_dirs:
        print("\nERROR: Some required files or directories are missing:")
        
        if missing_files:
            print("\nMissing files:")
            for file in missing_files:
                print(f"  - {file}")
        
        if missing_dirs:
            print("\nMissing directories:")
            for directory in missing_dirs:
                print(f"  - {directory}")
        
        print("\nPlease make sure all files and directories are in place before building.")
        return False
    
    # Check for main module files
    module_files = {
        "app/__init__.py": "Main app module",
        "app/core/__init__.py": "Core module",
        "app/agents/__init__.py": "Agents module",
        "app/ui/__init__.py": "UI module",
        "app/utils/__init__.py": "Utils module"
    }
    
    missing_modules = []
    for module_file, description in module_files.items():
        if not os.path.isfile(module_file):
            missing_modules.append((module_file, description))
    
    if missing_modules:
        print("\nWARNING: Some module initialization files are missing:")
        for module_file, description in missing_modules:
            print(f"  - {module_file} ({description})")
        print("This might cause import issues. Creating empty files...")
        
        # Create empty __init__.py files
        for module_file, _ in missing_modules:
            os.makedirs(os.path.dirname(module_file), exist_ok=True)
            with open(module_file, 'w') as f:
                f.write('"""' + os.path.dirname(module_file).replace('/', '.') + ' module"""')
            print(f"  Created {module_file}")
    
    # Check for config files
    if not os.path.isfile("config/config.json"):
        print("\nWARNING: config/config.json is missing.")
        if os.path.isfile("config/default_config.json"):
            print("  Found config/default_config.json, copying it to config/config.json...")
            import shutil
            shutil.copy("config/default_config.json", "config/config.json")
        else:
            print("  Creating a basic config/config.json...")
            os.makedirs("config", exist_ok=True)
            with open("config/config.json", 'w') as f:
                f.write('{\n    "api_keys": {\n        "huggingface": ""\n    },\n    "models": {\n        "default_model": "meta-llama/Llama-3-8B-Instruct",\n        "cache_dir": "~/.cache/sagax1/models"\n    },\n    "ui": {\n        "theme": "light",\n        "font_size": 12\n    },\n    "agents": {\n        "default_agent": "text_completion",\n        "max_history": 100\n    },\n    "execution": {\n        "python_executor": "local",\n        "max_execution_time": 30,\n        "authorized_imports": [\n            "numpy", "pandas", "matplotlib", "PIL", "requests", \n            "bs4", "datetime", "math", "re", "os", "csv", "json"\n        ]\n    }\n}')
    
    # Check for LICENSE.txt
    if not os.path.isfile("LICENSE.txt"):
        print("\nINFO: LICENSE.txt is missing. Creating a placeholder...")
        with open("LICENSE.txt", 'w') as f:
            f.write("sagax1 Software License\n\nCopyright (c) 2025 sagax1 Team\nAll rights reserved.\n")
    
    print("\nProject structure check completed successfully.")
    return True

if __name__ == "__main__":
    success = check_structure()
    sys.exit(0 if success else 1)