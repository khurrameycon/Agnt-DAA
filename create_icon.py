# Save this script as create_icon.py
import os
from PIL import Image

def create_ico_file():
    """Convert the app logo to an ICO file for Windows"""
    # Make sure the directory exists
    os.makedirs("assets/icons", exist_ok=True)
    
    # Check if we have a PNG logo
    png_path = r"F:\Freelancing\Ernest - Sagax\sagax1-smolagents\assets\icons\sagax1-logo.png"
    ico_path = r"F:\Freelancing\Ernest - Sagax\sagax1-smolagents\assets\icons\sagax1-logo.ico"
    
    # If the PNG exists but the ICO doesn't, convert it
    if os.path.exists(png_path) and not os.path.exists(ico_path):
        try:
            img = Image.open(png_path)
            
            # Create multiple sizes for the icon
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            img.save(ico_path, format='ICO', sizes=sizes)
            print(f"Created icon file at {ico_path}")
        except Exception as e:
            print(f"Error creating icon file: {str(e)}")
    elif not os.path.exists(png_path):
        print(f"Warning: Logo image not found at {png_path}")
        print("Please create a PNG logo file before building the installer")
    else:
        print(f"Icon file already exists at {ico_path}")

if __name__ == "__main__":
    create_ico_file()