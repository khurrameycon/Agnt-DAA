from smolagents import Tool
from gradio_client import Client
import tempfile
import os

# Test script for the RAG agent
def test_rag_pdf():
    # Create a test PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        temp_path = temp.name
        with open(r"C:\Users\Umair\Downloads\Receipt-2073-3086-3364.pdf", "rb") as f:
            temp.write(f.read())
        print(f"Test PDF at: {temp_path}")
    
    try:
        # Connect to space
        client = Client("cvachet/pdf-chatbot")
        
        # 1. Upload and process PDF
        print("Step 1: Uploading PDF")
        result = client.predict(
            [temp_path],  # List of files
            600,          # Chunk size
            40,           # Chunk overlap
            api_name="/initialize_database"
        )
        vector_db, collection_name, status = result
        print(f"Database initialization: {status}")
        
        # 2. Initialize QA chain
        print("Step 2: Setting up QA chain")
        qa_result = client.predict(
            0,          # LLM option (Mistral)
            0.7,        # Temperature
            1024,       # Max tokens
            3,          # Top-k
            vector_db,  # Vector DB from previous step
            api_name="/initialize_llm"
        )
        qa_chain, status = qa_result
        print(f"QA chain initialization: {status}")
        
        # 3. Ask a question
        print("Step 3: Asking question")
        question = "What is this document about?"
        history = []
        
        response = client.predict(
            qa_chain,   # QA chain
            question,   # Question
            history,    # Empty history
            api_name="/conversation"
        )
        
        # Parse response
        _, _, new_history, source1, page1, source2, page2, source3, page3 = response
        answer = new_history[-1][1] if new_history else "No answer"
        
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Source: {source1} (Page {page1})")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    test_rag_pdf()