"""
Test script for video generation with SahaniJi/Instant-Video space
"""

import os
import sys
import tempfile
from gradio_client import Client

def test_video_generation(prompt):
    """Test video generation with SahaniJi/Instant-Video space
    
    Args:
        prompt: Text prompt for video generation
    
    Returns:
        Path to generated video file
    """
    print(f"Generating video with prompt: {prompt}")
    
    try:
        # Connect to the space
        client = Client("SahaniJi/Instant-Video")
        
        print("Connected to SahaniJi/Instant-Video")
        
        # Based on the output we can see the API endpoints
        # Instead of relying on view_api(), we'll use the known endpoint names
        api_name = "/instant_video"  # This appears to be the correct endpoint
        
        print(f"Using API endpoint: {api_name}")
        
        # Set default parameters based on the observed API
        base = "Realistic"
        motion = "guoyww/animatediff-motion-lora-zoom-in"
        step = "4"
        
        print("Sending request to generate video...")
        print(f"Parameters: prompt='{prompt}', base='{base}', motion='{motion}', step='{step}'")
        
        # Generate video with parameters
        result = client.predict(
            prompt,
            base,
            motion,
            step,
            api_name=api_name
        )
        
        print(f"Result type: {type(result)}")
        
        # If result is a dictionary or object with attributes, inspect it more deeply
        if hasattr(result, '__dict__'):
            print(f"Result attributes: {dir(result)}")
        
        if isinstance(result, dict):
            print(f"Result dictionary keys: {result.keys()}")
            
            # Check for 'video' key which appears to be the expected format
            if 'video' in result:
                video_path = result['video']
                print(f"Found video path in result dictionary: {video_path}")
                
                # Check file size
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    print(f"Video file size: {file_size} bytes")
                    return video_path
                else:
                    print(f"Video file not found at path: {video_path}")
        
        # If result is a list or tuple, print each item
        elif isinstance(result, (list, tuple)):
            print(f"Result is a {type(result).__name__} with {len(result)} items")
            for i, item in enumerate(result):
                print(f"Item {i}: {type(item)} - {item}")
                
                # Check if this item is a string and looks like a file path
                if isinstance(item, str) and (item.endswith('.mp4') or item.endswith('.webm') or item.endswith('.avi')):
                    if os.path.exists(item):
                        print(f"Found video file: {item}")
                        return item
        
        # If we don't understand the result format, save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = temp_file.name
            print(f"Saving result to temporary file: {video_path}")
            
            if isinstance(result, bytes):
                temp_file.write(result)
            elif hasattr(result, 'read'):
                temp_file.write(result.read())
            elif isinstance(result, str):
                temp_file.write(result.encode('utf-8'))
            else:
                # Just write the string representation
                temp_file.write(str(result).encode('utf-8'))
        
        # Check file size
        file_size = os.path.getsize(video_path)
        print(f"Temporary file size: {file_size} bytes")
        
        return video_path
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Get prompt from command line arguments or use default
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A lady using a mobile phone and smiling"
    
    print("=" * 50)
    print(f"Testing video generation with prompt: {prompt}")
    print("=" * 50)
    
    video_path = test_video_generation(prompt)
    
    if video_path:
        print(f"Video generated at: {video_path}")
        
        # Check file size again
        file_size = os.path.getsize(video_path)
        print(f"Final video file size: {file_size} bytes")
        
        if file_size < 1000:
            print(f"Warning: Video file is suspiciously small ({file_size} bytes)")
            
            # Let's examine the file content
            with open(video_path, 'rb') as f:
                content = f.read(100)  # Read first 100 bytes
                print(f"File content (first 100 bytes): {content}")
        
        # Try to play the video automatically
        import platform
        import subprocess
        
        system = platform.system()
        try:
            if system == 'Windows':
                os.startfile(video_path)
            elif system == 'Darwin':  # macOS
                subprocess.call(['open', video_path])
            else:  # Linux
                subprocess.call(['xdg-open', video_path])
            
            print(f"Opened video with default player")
        except Exception as e:
            print(f"Could not open video automatically: {str(e)}")
            print(f"Please open the video manually at: {video_path}")
    else:
        print("Video generation failed")