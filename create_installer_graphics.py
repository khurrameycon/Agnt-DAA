# Save as create_installer_graphics.py
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

def create_installer_graphics():
    """Create graphics needed for the NSIS installer"""
    # Make sure the directory exists
    os.makedirs("assets/icons", exist_ok=True)
    
    # Check if we have a logo
    logo_path = "assets/icons/sagax1-logo.png"
    if not os.path.exists(logo_path):
        print(f"Warning: Logo not found at {logo_path}")
        print("Creating a simple logo...")
        create_simple_logo()
    
    # Create the header bitmap (150x57 pixels)
    create_header_bitmap()
    
    # Create the welcome bitmap (164x314 pixels)
    create_welcome_bitmap()
    
    print("Installer graphics created successfully")

def create_simple_logo():
    """Create a simple logo if none exists"""
    # Create a 256x256 image with a blue background
    img = Image.new('RGB', (256, 256), color=(45, 95, 139))  # #2D5F8B blue
    
    # Draw text on it
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except:
        font = ImageFont.load_default()
    
    # Draw S1 text centered
    draw.text((128, 100), "S1", fill=(255, 255, 255), font=font, anchor="mm")
    
    # Save the image
    img.save("assets/icons/sagax1-logo.png")
    print("Created simple logo at assets/icons/sagax1-logo.png")

def create_header_bitmap():
    """Create the header bitmap for the installer"""
    # Header bitmap is 150x57 pixels
    header_path = "assets/icons/installer_header.bmp"
    
    # Check if the file already exists
    if os.path.exists(header_path):
        print(f"Header bitmap already exists at {header_path}")
        return
    
    # Create a new image with gradient background
    img = Image.new('RGB', (150, 57), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw a gradient background
    for y in range(57):
        r = int(45 + (y / 57) * 40)
        g = int(95 + (y / 57) * 40)
        b = int(139 + (y / 57) * 40)
        draw.line([(0, y), (150, y)], fill=(r, g, b))
    
    # Load the logo
    try:
        logo = Image.open("assets/icons/sagax1-logo.png")
        # Resize maintaining aspect ratio
        logo.thumbnail((40, 40))
        
        # Paste the logo on the left
        img.paste(logo, (10, 8), logo if logo.mode == 'RGBA' else None)
    except Exception as e:
        print(f"Couldn't add logo to header: {e}")
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        draw.text((60, 20), "sagax1", fill=(255, 255, 255), font=font)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        draw.text((60, 20), "sagax1", fill=(255, 255, 255))
    
    # Save as BMP
    img.save(header_path, format="BMP")
    print(f"Created header bitmap at {header_path}")

def create_welcome_bitmap():
    """Create the welcome bitmap for the installer"""
    # Welcome bitmap is 164x314 pixels
    welcome_path = "assets/icons/installer_welcome.bmp"
    
    # Check if the file already exists
    if os.path.exists(welcome_path):
        print(f"Welcome bitmap already exists at {welcome_path}")
        return
    
    # Create a new image with gradient background
    img = Image.new('RGB', (164, 314), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw a gradient background
    for y in range(314):
        r = int(45 + (y / 314) * 70)
        g = int(95 + (y / 314) * 70)
        b = int(139 + (y / 314) * 70)
        draw.line([(0, y), (164, y)], fill=(r, g, b))
    
    # Load the logo
    try:
        logo = Image.open("assets/icons/sagax1-logo.png")
        # Resize maintaining aspect ratio
        logo.thumbnail((100, 100))
        
        # Center the logo
        x = (164 - logo.width) // 2
        img.paste(logo, (x, 80), logo if logo.mode == 'RGBA' else None)
    except Exception as e:
        print(f"Couldn't add logo to welcome bitmap: {e}")
    
    # Add title text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        title_width = draw.textlength("sagax1", font=font)
        draw.text(((164 - title_width) // 2, 190), "sagax1", fill=(255, 255, 255), font=font)
        
        # Add subtitle text
        font_small = ImageFont.truetype("arial.ttf", 10)
        subtitle = "Opensource AI-powered agent platform"
        subtitle_width = draw.textlength(subtitle, font=font_small)
        draw.text(((164 - subtitle_width) // 2, 220), subtitle, fill=(220, 220, 220), font=font_small)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        draw.text((50, 190), "sagax1", fill=(255, 255, 255))
        draw.text((30, 220), "Opensource AI-powered agent platform", fill=(220, 220, 220))
    
    # Save as BMP
    img.save(welcome_path, format="BMP")
    print(f"Created welcome bitmap at {welcome_path}")

if __name__ == "__main__":
    create_installer_graphics()