"""
Test script for code generation with Hugging Face spaces
"""

import os
import sys
import tempfile
import time
from gradio_client import Client

def test_code_generation(prompt):
    """Test code generation with Hugging Face spaces
    
    Args:
        prompt: Text prompt for code generation
    
    Returns:
        Generated code
    """
    print(f"Generating code with prompt: {prompt}")
    
    try:
        # Connect to the primary space
        space_id = "sitammeur/Qwen-Coder-llamacpp"
        print(f"Connecting to space: {space_id}")
        
        client = Client(space_id)
        
        # Print API information if available, but don't fail if it's None
        try:
            api_info = client.view_api()
            if api_info:
                print("Available APIs:")
                for name, info in api_info.items():
                    print(f"- {name}: {info['parameters']}")
        except Exception as api_error:
            print(f"Could not retrieve API info: {str(api_error)}")
        
        # Based on the output, we need to use the "/chat" API endpoint
        print("Using '/chat' API endpoint")
        
        # Generate code using the parameters from the output
        print("Sending request to generate code...")
        result = client.predict(
            prompt,  # message parameter
            "Qwen2.5-Coder-0.5B-Instruct-Q6_K.gguf",  # model parameter
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that generates code.",  # system prompt
            1024,  # max_length
            0.7,   # temperature
            0.95,  # top_p
            40,    # frequency_penalty
            1.1,   # presence_penalty
            api_name="/chat"
        )
        
        print(f"Result type: {type(result)}")
        
        # Check result
        if isinstance(result, dict):
            print("Response is a dictionary with keys:", result.keys())
            # Extract code from the response based on the structure
            if "response" in result:
                code = result["response"]
                print("Response preview:")
                print(code[:500] + ("..." if len(code) > 500 else ""))
                return code
            else:
                print(f"Unexpected response structure: {result}")
                return str(result)
        elif isinstance(result, str):
            print("Response preview:")
            print(result[:500] + ("..." if len(result) > 500 else ""))
            return result
        elif isinstance(result, (list, tuple)):
            print(f"Result is a {type(result).__name__} with {len(result)} items")
            for i, item in enumerate(result):
                print(f"Item {i}: {type(item)} - {str(item)[:100]}")
            return str(result[0]) if result else "No result"
        else:
            print(f"Unexpected result type: {type(result)}")
            return str(result)
    
    except Exception as e:
        import traceback
        print(f"Error with primary space: {str(e)}")
        traceback.print_exc()
        
        # Try fallback space
        try:
            print("\nTrying fallback space...")
            fallback_space = "Pradipta01/Code_Generator"
            print(f"Connecting to fallback space: {fallback_space}")
            
            client = Client(fallback_space)
            
            # For the fallback space, we need to use the default API endpoint
            try:
                api_info = client.view_api()
                if api_info:
                    print("Fallback API info:")
                    for name, info in api_info.items():
                        print(f"- {name}")
            except Exception as api_error:
                print(f"Could not retrieve fallback API info: {str(api_error)}")
            
            # Try the direct interface without API name
            print("Sending request to fallback space...")
            # Check if the first parameter is the prompt or a different parameter
            result = client.predict(prompt)
            
            if isinstance(result, str):
                print("Fallback result preview:")
                print(result[:500] + ("..." if len(result) > 500 else ""))
                return result
            else:
                print(f"Unexpected fallback result type: {type(result)}")
                return str(result)
        except Exception as fallback_error:
            print(f"Fallback error: {str(fallback_error)}")
            
            # Try one more fallback space
            try:
                print("\nTrying second fallback space: microsoft/CodeX-code-generation")
                client = Client("microsoft/CodeX-code-generation")
                result = client.predict(prompt)
                print("Second fallback result preview:")
                print(str(result)[:500] + ("..." if len(str(result)) > 500 else ""))
                return str(result)
            except Exception as second_fallback_error:
                print(f"Second fallback error: {str(second_fallback_error)}")
                
                return f"Failed to generate code with all spaces."

if __name__ == "__main__":
    # Get prompt from command line arguments or use default
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a Python function to find the longest palindrome in a string"
    
    print("=" * 50)
    print(f"Testing code generation with prompt: {prompt}")
    print("=" * 50)
    
    generated_code = test_code_generation(prompt)
    
    if generated_code:
        # Save the generated code to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(generated_code)
            temp_path = temp_file.name
            
        print(f"\nGenerated code saved to: {temp_path}")
        
        # Try to open the file with default editor
        try:
            import platform
            import subprocess
            
            system = platform.system()
            if system == 'Windows':
                os.startfile(temp_path)
            elif system == 'Darwin':  # macOS
                subprocess.call(['open', temp_path])
            else:  # Linux
                subprocess.call(['xdg-open', temp_path])
            
            print(f"Opened code file with default editor")
        except Exception as e:
            print(f"Could not open code file automatically: {str(e)}")
            print(f"Please open the file manually at: {temp_path}")
    else:
        print("No code was generated")