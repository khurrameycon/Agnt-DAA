"""
RAG Agent for sagax1
Agent for document retrieval and question answering using Hugging Face spaces
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Callable, Union
import base64
import json
import time
import re
from pathlib import Path

from app.agents.base_agent import BaseAgent
from gradio_client import Client

class RagAgent(BaseAgent):
    """Agent for document retrieval and question answering"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the RAG agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
                model_id: Hugging Face model ID (optional, for compatibility)
                rag_space_id: Primary Hugging Face space ID for RAG
                fallback_space_id: Fallback Hugging Face space ID for RAG
        """
        super().__init__(agent_id, config)
        
        # Store configuration
        self.rag_space_id = config.get("rag_space_id", "AdrienB134/rag_ColPali_Qwen2VL")
        self.fallback_space_id = config.get("fallback_space_id", "openfree/PDF-RAG")
        
        # Initialize clients
        self.primary_client = None
        self.fallback_client = None
        self.active_client = None
        self.active_space_id = None
        
        # Track uploaded documents
        self.documents = {}  # Mapping of document ID to file path
        self.is_initialized = False
        
        # Additional tracking
        self.last_query = ""
        self.latest_document_id = None
    
    def initialize(self) -> None:
        """Initialize the RAG clients with fallback to PDF-RAG"""
        if self.is_initialized:
            return
        
        try:
            # Skip ColPali since it's causing errors and go straight to PDF-RAG
            self.logger.info(f"Initializing RAG agent with fallback space {self.fallback_space_id}")
            
            try:
                self.logger.info(f"Connecting to PDF-RAG space: {self.fallback_space_id}")
                self.fallback_client = Client(self.fallback_space_id)
                self.active_client = self.fallback_client
                self.active_space_id = self.fallback_space_id
                self.logger.info("Successfully connected to PDF-RAG space")
            except Exception as e:
                self.logger.error(f"Failed to connect to PDF-RAG space: {str(e)}")
                self.fallback_client = None
                raise RuntimeError("Could not connect to any RAG space")
            
            self.is_initialized = True
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG agent: {str(e)}")
            raise

    
    def upload_document(self, file_path: str, callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Upload a document to the RAG space
        
        Args:
            file_path: Path to the document file
            callback: Optional callback for progress updates
            
        Returns:
            Dict with document info or error message
        """
        if not self.is_initialized:
            self.initialize()
        
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()
            
            if callback:
                callback(f"Uploading document: {file_name}...")
            
            # Different handling based on active space
            if self.active_space_id == self.rag_space_id:
                # AdrienB134/rag_ColPali_Qwen2VL space
                return self._upload_to_colpali_space(file_path, file_name, callback)
            elif self.active_space_id == self.fallback_space_id:
                # openfree/PDF-RAG space
                return self._upload_to_pdf_rag_space(file_path, file_name, callback)
            else:
                error_msg = "No active RAG space available"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error uploading document: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _upload_to_colpali_space(self, file_path: str, file_name: str, 
                           callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Upload document to ColPali RAG space
        
        Args:
            file_path: Path to document file
            file_name: Name of the file
            callback: Optional progress callback
            
        Returns:
            Upload result
        """
        try:
            if callback:
                callback("Processing document...")
            
            # Generate a document ID
            document_id = f"doc_{int(time.time())}_{os.path.splitext(file_name)[0]}"
            
            # Store document path
            self.documents[document_id] = file_path
            self.latest_document_id = document_id
            
            # First, try to get API info to see available functions
            try:
                api_info = self.primary_client.view_api()
                self.logger.info(f"Available API endpoints: {api_info}")
                
                if callback:
                    callback("Found API endpoints, attempting upload...")
            except Exception as api_error:
                self.logger.warning(f"Could not retrieve API info: {str(api_error)}")
            
            # Try with function indices instead of API names
            fn_indices = [0, 1, 2, 3, 4, 5]  # Try the first 6 function indices
            result = None
            success_index = None
            
            for idx in fn_indices:
                try:
                    self.logger.info(f"Trying to upload document with fn_index {idx}")
                    if callback:
                        callback(f"Trying to upload with index {idx}...")
                        
                    result = self.primary_client.predict(
                        file_path,  # PDF file
                        fn_index=idx
                    )
                    success_index = idx
                    self.logger.info(f"Upload successful with fn_index {idx}")
                    break
                except Exception as upload_error:
                    self.logger.warning(f"Upload failed with fn_index {idx}: {str(upload_error)}")
            
            if result is None:
                # Try submitting to the client directly
                try:
                    self.logger.info("Trying client.submit() method")
                    if callback:
                        callback("Trying alternative upload method...")
                    
                    # Try to use client's submit method which doesn't require endpoint specification
                    for idx in fn_indices:
                        try:
                            job = self.primary_client.submit(file_path, fn_index=idx)
                            result = job.result()
                            success_index = f"submit_{idx}"
                            self.logger.info(f"Upload successful with client.submit() and fn_index {idx}")
                            break
                        except Exception as submit_error:
                            self.logger.warning(f"Submit failed with fn_index {idx}: {str(submit_error)}")
                except Exception as submit_error:
                    self.logger.error(f"All submit attempts failed: {str(submit_error)}")
            
            if result is None:
                # If all attempts failed, raise exception
                raise Exception("Could not upload document with any method")
            
            if callback:
                callback(f"Document uploaded successfully: {file_name}")
            
            return {
                "success": True,
                "document_id": document_id,
                "file_name": file_name,
                "message": f"Document uploaded successfully: {file_name}",
                "method": f"fn_index_{success_index}" if success_index is not None else "unknown",
                "raw_result": result
            }
            
        except Exception as e:
            error_msg = f"Error uploading to ColPali space: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _upload_to_pdf_rag_space(self, file_path: str, file_name: str,
                           callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Upload document to PDF RAG space using fn_index instead of API names
        
        Args:
            file_path: Path to document file
            file_name: Name of the file
            callback: Optional progress callback
            
        Returns:
            Upload result
        """
        try:
            if callback:
                callback("Processing PDF document...")
            
            # Generate a document ID
            document_id = f"doc_{int(time.time())}_{os.path.splitext(file_name)[0]}"
            
            # Store document path
            self.documents[document_id] = file_path
            self.latest_document_id = document_id
            
            # Try to get API info
            try:
                api_info = self.fallback_client.view_api()
                self.logger.info(f"PDF-RAG API endpoints: {api_info}")
            except Exception as api_error:
                self.logger.warning(f"Could not retrieve PDF-RAG API info: {str(api_error)}")
            
            # Try specific fn_index values (0 is often for file upload)
            fn_indices = [0, 1, 2]  # Try first few indices
            result = None
            success_index = None
            
            for idx in fn_indices:
                try:
                    self.logger.info(f"Trying to upload PDF with fn_index {idx}")
                    if callback:
                        callback(f"Trying to upload PDF with index {idx}...")
                        
                    result = self.fallback_client.predict(
                        file_path,  # PDF file
                        fn_index=idx
                    )
                    success_index = idx
                    self.logger.info(f"Upload successful with fn_index {idx}")
                    break
                except Exception as upload_error:
                    self.logger.warning(f"Upload failed with fn_index {idx}: {str(upload_error)}")
            
            if result is None:
                # Try simple approach - if the space has a single obvious upload function
                try:
                    self.logger.info("Trying simple predict with no index")
                    result = self.fallback_client.predict(file_path)
                    success_index = "no_index"
                    self.logger.info("Upload successful with no index")
                except Exception as e:
                    self.logger.warning(f"Simple predict failed: {str(e)}")
                    
                    # Try with file parameter name
                    try:
                        self.logger.info("Trying with 'file' parameter name")
                        result = self.fallback_client.predict(
                            file=file_path,
                            fn_index=0
                        )
                        success_index = "file_param"
                        self.logger.info("Upload successful with 'file' parameter")
                    except Exception as e:
                        self.logger.warning(f"File parameter approach failed: {str(e)}")
            
            if result is None:
                # If all attempts failed, raise exception
                raise Exception("Could not upload document with any method")
            
            if callback:
                callback(f"PDF document uploaded successfully: {file_name}")
            
            return {
                "success": True,
                "document_id": document_id,
                "file_name": file_name,
                "message": f"PDF document uploaded successfully: {file_name}",
                "method": f"fn_index_{success_index}" if success_index is not None else "unknown",
                "raw_result": result
            }
            
        except Exception as e:
            error_msg = f"Error uploading to PDF-RAG space: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def query_document(self, query: str, document_id: Optional[str] = None,
                    callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Query a document with a question
        
        Args:
            query: Question to ask about the document
            document_id: ID of the document to query (if None, uses latest)
            callback: Optional callback for progress updates
            
        Returns:
            Query response
        """
        if not self.is_initialized:
            self.initialize()
        
        if not self.active_client:
            error_msg = "No active RAG space available"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            if callback:
                callback(f"Processing query: {query}")
            
            # Store the query
            self.last_query = query
            
            # Use provided document ID or latest one
            doc_id = document_id or self.latest_document_id
            
            if not doc_id or doc_id not in self.documents:
                error_msg = "No document available for querying"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Different handling based on active space
            if self.active_space_id == self.rag_space_id:
                # AdrienB134/rag_ColPali_Qwen2VL space
                return self._query_colpali_space(query, doc_id, callback)
            elif self.active_space_id == self.fallback_space_id:
                # openfree/PDF-RAG space
                return self._query_pdf_rag_space(query, doc_id, callback)
            else:
                error_msg = "No active RAG space available"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error querying document: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _query_colpali_space(self, query: str, document_id: str,
                      callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Query document using ColPali RAG space
        
        Args:
            query: Question to ask
            document_id: Document ID
            callback: Optional progress callback
            
        Returns:
            Query result
        """
        try:
            if callback:
                callback("Processing query with ColPali space...")
            
            # Try with function indices instead of API names
            fn_indices = [0, 1, 2, 3, 4, 5]  # Try the first 6 function indices
            result = None
            success_index = None
            
            for idx in fn_indices:
                try:
                    self.logger.info(f"Trying to query with fn_index {idx}")
                    if callback:
                        callback(f"Trying to query with index {idx}...")
                        
                    result = self.primary_client.predict(
                        query,  # The question
                        fn_index=idx
                    )
                    success_index = idx
                    self.logger.info(f"Query successful with fn_index {idx}")
                    break
                except Exception as query_error:
                    self.logger.warning(f"Query failed with fn_index {idx}: {str(query_error)}")
            
            if result is None:
                # Try submitting to the client directly
                try:
                    self.logger.info("Trying client.submit() method for query")
                    if callback:
                        callback("Trying alternative query method...")
                    
                    # Try to use client's submit method
                    for idx in fn_indices:
                        try:
                            job = self.primary_client.submit(query, fn_index=idx)
                            result = job.result()
                            success_index = f"submit_{idx}"
                            self.logger.info(f"Query successful with client.submit() and fn_index {idx}")
                            break
                        except Exception as submit_error:
                            self.logger.warning(f"Submit query failed with fn_index {idx}: {str(submit_error)}")
                except Exception as submit_error:
                    self.logger.error(f"All submit query attempts failed: {str(submit_error)}")
            
            if result is None:
                # If all attempts failed, raise exception
                raise Exception("Could not query with any method")
            
            # Process the result
            if isinstance(result, str):
                answer = result
            elif isinstance(result, dict):
                answer = result.get("answer", str(result))
            else:
                answer = str(result)
            
            # Add to history
            self.add_to_history(query, answer)
            
            if callback:
                callback("Query processed successfully")
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "document_id": document_id,
                "method": f"fn_index_{success_index}" if success_index is not None else "unknown",
                "raw_result": result
            }
            
        except Exception as e:
            error_msg = f"Error querying ColPali space: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _query_pdf_rag_space(self, query: str, document_id: str,
                      callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Query document using PDF RAG space
        
        Args:
            query: Question to ask
            document_id: Document ID
            callback: Optional progress callback
            
        Returns:
            Query result
        """
        try:
            if callback:
                callback("Processing query with PDF-RAG space...")
            
            # Try with various function indices
            fn_indices = [0, 1, 2, 3]  # Try the first few function indices
            result = None
            success_index = None
            
            for idx in fn_indices:
                try:
                    self.logger.info(f"Trying to query with fn_index {idx}")
                    if callback:
                        callback(f"Trying to query with index {idx}...")
                        
                    result = self.fallback_client.predict(
                        query,  # The question
                        fn_index=idx
                    )
                    success_index = idx
                    self.logger.info(f"Query successful with fn_index {idx}")
                    break
                except Exception as query_error:
                    self.logger.warning(f"Query failed with fn_index {idx}: {str(query_error)}")
            
            if result is None:
                # Try with likely parameter names
                try:
                    self.logger.info("Trying with 'question' parameter name")
                    result = self.fallback_client.predict(
                        question=query,
                        fn_index=1  # Often query is the second function
                    )
                    success_index = "question_param"
                    self.logger.info("Query successful with 'question' parameter")
                except Exception as e:
                    self.logger.warning(f"Question parameter approach failed: {str(e)}")
                    
                    # Try with text parameter
                    try:
                        self.logger.info("Trying with 'text' parameter name")
                        result = self.fallback_client.predict(
                            text=query,
                            fn_index=1
                        )
                        success_index = "text_param"
                        self.logger.info("Query successful with 'text' parameter")
                    except Exception as e:
                        self.logger.warning(f"Text parameter approach failed: {str(e)}")
            
            if result is None:
                # If all attempts failed, raise exception
                raise Exception("Could not query with any method")
            
            # Process the result
            if isinstance(result, str):
                answer = result
            elif isinstance(result, dict):
                answer = result.get("answer", str(result))
            else:
                answer = str(result)
            
            # Add to history
            self.add_to_history(query, answer)
            
            if callback:
                callback("Query processed successfully")
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "document_id": document_id,
                "method": f"fn_index_{success_index}" if success_index is not None else "unknown",
                "raw_result": result
            }
            
        except Exception as e:
            error_msg = f"Error querying PDF-RAG space: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input
        
        Args:
            input_text: Input text for the agent (query)
            callback: Optional callback for streaming responses
            
        Returns:
            Agent output text
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Check if input is a command in JSON format
            try:
                command = json.loads(input_text)
                action = command.get("action", "")
                
                if action == "upload":
                    # Upload document
                    file_path = command.get("file_path", "")
                    result = self.upload_document(file_path, callback)
                    return json.dumps(result, indent=2)
                elif action == "query":
                    # Query document
                    query = command.get("query", "")
                    document_id = command.get("document_id")
                    result = self.query_document(query, document_id, callback)
                    return json.dumps(result, indent=2)
                else:
                    # Unknown action
                    error_msg = f"Unknown action: {action}"
                    self.logger.error(error_msg)
                    return f"Error: {error_msg}"
            except json.JSONDecodeError:
                # Not JSON, treat as regular query
                # If we have a document uploaded, query it
                if self.latest_document_id:
                    result = self.query_document(input_text, self.latest_document_id, callback)
                    if result["success"]:
                        return result["answer"]
                    else:
                        return f"Error: {result.get('error', 'Unknown error')}"
                else:
                    return "Please upload a document first."
                
        except Exception as e:
            error_msg = f"Error in RAG agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error: {error_msg}"
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
        self.documents = {}
        self.latest_document_id = None
        self.last_query = ""
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        return [
            "document_retrieval",
            "question_answering", 
            "document_upload",
            "document_search"
        ]
    
    def cleanup(self) -> None:
        """Clean up resources"""
        # Close any open connections
        self.primary_client = None
        self.fallback_client = None
        self.active_client = None
        
        # Reset state
        self.reset()