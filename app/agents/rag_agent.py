"""
RAG Agent for sagax1
Agent for document retrieval and question answering using FAISS vector store
Updated to use API providers (Groq, OpenAI, Gemini) instead of HuggingFace
"""
import sys
import os
import logging
import tempfile
import uuid
import json
import re
import concurrent.futures
import threading
import time
import gc
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union, Generator
from collections import OrderedDict

from app.agents.base_agent import BaseAgent

# Import necessary libraries for document processing
from unidecode import unidecode
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import necessary libraries for vector database
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# Import necessary libraries for LLM - Remove HF dependencies
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from app.core.config_manager import ConfigManager
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Resource monitoring and limitation
MAX_MEMORY_PERCENT = 80  # Maximum memory usage percentage
MAX_CPU_PERCENT = 70     # Maximum CPU usage percentage
CHECK_RESOURCE_INTERVAL = 2.0  # Check resources every 2 seconds

# LRU Cache implementation for memory management
class LRUCache:
    """LRU Cache implementation with size limit"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        # Move accessed item to the end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        # If key exists, update and move to end
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
            return
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        
        # Add new item
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
    
    def size(self):
        return len(self.cache)

# Resource monitoring thread
class ResourceMonitor:
    """Monitor system resources and adjust processing accordingly"""
    def __init__(self, max_memory_percent=MAX_MEMORY_PERCENT, 
                 max_cpu_percent=MAX_CPU_PERCENT,
                 check_interval=CHECK_RESOURCE_INTERVAL):
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self.check_interval = check_interval
        self.should_stop = threading.Event()
        self.paused = threading.Event()
        self.monitor_thread = None
    
    def start(self):
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.should_stop.clear()
            self.paused.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop(self):
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.should_stop.set()
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        while not self.should_stop.is_set():
            # Check memory usage
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # If resources are above thresholds, pause processing
            if memory_percent > self.max_memory_percent or cpu_percent > self.max_cpu_percent:
                self.paused.set()
                # Force garbage collection when resources are constrained
                gc.collect()
            else:
                self.paused.clear()
            
            # Sleep for a bit
            time.sleep(self.check_interval)

class RAGAgent(BaseAgent):
    """Agent for document retrieval and question answering using FAISS vector store"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the RAG agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
        """
        super().__init__(agent_id, config)
        
        # Get config parameters or use defaults
        self.model_id = config.get("model_id", "llama-3.3-70b-versatile")  # Default to Groq model
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.top_k = config.get("top_k", 3)
        
        # API provider configuration
        self.api_provider = config.get("api_provider", "groq")
        
        # Advanced chunking parameters
        self.chunk_size = config.get("chunk_size", 600)
        self.chunk_overlap = config.get("chunk_overlap", 40)
        self.use_dynamic_chunking = config.get("use_dynamic_chunking", True)
        self.min_chunk_size = config.get("min_chunk_size", 300)
        self.max_chunk_size = config.get("max_chunk_size", 800)
        
        # Hybrid search parameters
        self.use_hybrid_search = config.get("use_hybrid_search", True)
        self.semantic_weight = config.get("semantic_weight", 0.7)
        self.keyword_weight = config.get("keyword_weight", 0.3)
        
        # MMR parameters
        self.use_mmr = config.get("use_mmr", True)
        self.mmr_lambda = config.get("mmr_lambda", 0.7)
        
        # Parallel processing - using threads instead of processes to prevent window spawning
        self.use_parallel = config.get("use_parallel", False)
        cpu_count = os.cpu_count() or 1
        self.max_workers = config.get("max_workers", max(1, min(cpu_count - 2, 6)))
        
        # Resource monitoring and limitations
        self.max_memory_percent = config.get("max_memory_percent", MAX_MEMORY_PERCENT)
        self.max_cpu_percent = config.get("max_cpu_percent", MAX_CPU_PERCENT)
        
        # Cancellation support
        self.cancellation_token = threading.Event()
        
        # Set base directory for storing FAISS indexes
        self.faiss_index_dir = config.get("faiss_index_dir", "./faiss_indexes")
        os.makedirs(self.faiss_index_dir, exist_ok=True)
        
        # Status and components
        self.llm = None
        self.documents = {}  # Track document metadata (not full content)
        self.current_document_id = None
        self.is_initialized = False
        
        # Caching with size limits
        embedding_cache_size = config.get("embedding_cache_size", 10000)
        self.embedding_cache = LRUCache(embedding_cache_size)
        self.tfidf_vectorizers = {}
        
        # Resource monitor
        self.resource_monitor = ResourceMonitor(
            max_memory_percent=self.max_memory_percent,
            max_cpu_percent=self.max_cpu_percent
        )
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # System prompt template with context window optimization prompt
        self.SYSTEM_PROMPT = """
You are an assistant for question-answering tasks. Use the following pieces of context to answer the question at the end.
The context contains the most relevant information from a document that the user has provided.

When answering:
1. If the context directly answers the question, provide a clear and concise response.
2. If you need to synthesize information across different parts of the context, do so carefully.
3. If the question cannot be answered from the context, say "I don't have enough information to answer this question." 
4. Do not make up or infer information that's not at least implied by the context.
5. If appropriate, cite specific pages from the document mentioned in the context.

Question: {question} 
Context: {context} 
Helpful Answer:
"""
        self.logger.info(f"RAG Agent initialized with API provider: {self.api_provider}")
    
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        try:
            # Stop resource monitoring
            if hasattr(self, 'resource_monitor'):
                self.resource_monitor.stop()
            
            # Clean up LLM resources
            if hasattr(self, 'llm') and self.llm is not None:
                del self.llm
            
            # Clear caches
            if hasattr(self, 'embedding_cache'):
                self.embedding_cache.clear()
            
            # Force garbage collection
            gc.collect()
        except Exception as e:
            # Just log errors during cleanup, don't raise
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during RAGAgent cleanup: {str(e)}")
    
    def initialize(self) -> None:
        """Initialize embedding model with SSL fixes and proper authentication"""
        if self.is_initialized:
            return
        
        import sys
        
        self.logger.info(f"==== RAG AGENT INITIALIZATION START ====")
        
        # Fix SSL/HTTPS issues which are common in packaged applications
        try:
            import ssl
            import certifi
            import requests
            
            self.logger.info("Configuring SSL certificates for HTTPS requests")
            
            # Get the certifi certificate path
            cert_path = certifi.where()
            self.logger.info(f"Certifi certificate path: {cert_path}")
            
            # Set environment variables for certificates
            os.environ['SSL_CERT_FILE'] = cert_path
            os.environ['REQUESTS_CA_BUNDLE'] = cert_path
            os.environ['CURL_CA_BUNDLE'] = cert_path
            
            # For packaged apps, consider creating an unverified context
            if getattr(sys, 'frozen', False):
                self.logger.warning("Using unverified SSL context in packaged app")
                ssl._create_default_https_context = ssl._create_unverified_context
                
        except Exception as ssl_e:
            self.logger.error(f"Error setting up SSL certificates: {ssl_e}")
        
        # Try to use a bundled model first (for packaged applications)
        if getattr(sys, 'frozen', False):
            try:
                base_dir = os.path.dirname(sys.executable)
                model_path = os.path.join(base_dir, "models", "sentence_transformers", "all-MiniLM-L6-v2")
                
                self.logger.info(f"Checking for bundled model at: {model_path}")
                if os.path.exists(model_path):
                    self.logger.info(f"Using bundled model at: {model_path}")
                    self.embedding_model = HuggingFaceEmbeddings(
                        model_name=model_path,
                        cache_folder=os.path.join(base_dir, "models", "cache")
                    )
                    self.is_initialized = True
                    self.logger.info("RAG agent initialized successfully with bundled model")
                    return
                else:
                    self.logger.warning(f"Bundled model not found at: {model_path}, will try online model")
            except Exception as bundle_e:
                self.logger.error(f"Error trying to use bundled model: {bundle_e}")
        
        # If we get here, try online model
        try:
            # Set up the cache directory
            cache_dir = os.path.join(os.getcwd(), "models", "cache")
            if getattr(sys, 'frozen', False):
                cache_dir = os.path.join(os.path.dirname(sys.executable), "models", "cache")
            
            os.makedirs(cache_dir, exist_ok=True)
            self.logger.info(f"Using cache directory: {cache_dir}")
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            
            # Now try to initialize the embedding model
            try:
                self.logger.info("Initializing embedding model with all-MiniLM-L6-v2")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder=cache_dir,
                )
                
                self.is_initialized = True
                self.logger.info("RAG agent initialized successfully with online model")
                
            except Exception as model_e:
                self.logger.error(f"Error initializing with standard model: {model_e}")
                
                # Try with a different model
                try:
                    self.logger.info("Trying fallback model: distilbert-base-nli-mean-tokens")
                    self.embedding_model = HuggingFaceEmbeddings(
                        model_name="distilbert-base-nli-mean-tokens",
                        cache_folder=cache_dir,
                    )
                    
                    self.is_initialized = True
                    self.logger.info("RAG agent initialized successfully with fallback model")
                    
                except Exception as fallback_e:
                    self.logger.error(f"Error with fallback model: {fallback_e}")
                    
                    # As a last resort, create a simple embedding model
                    try:
                        self.logger.warning("Using simple TF-IDF embeddings as final fallback")
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        
                        class SimpleTfidfEmbeddings:
                            def __init__(self):
                                self.vectorizer = TfidfVectorizer(max_features=768)
                                self.is_fit = False
                                
                            def embed_documents(self, texts):
                                if not self.is_fit:
                                    self.vectorizer.fit(texts)
                                    self.is_fit = True
                                
                                matrix = self.vectorizer.transform(texts).toarray()
                                return [vector.tolist() for vector in matrix]
                                
                            def embed_query(self, text):
                                if not self.is_fit:
                                    self.vectorizer.fit([text])
                                    self.is_fit = True
                                    
                                return self.vectorizer.transform([text]).toarray()[0].tolist()
                        
                        self.embedding_model = SimpleTfidfEmbeddings()
                        self.is_initialized = True
                        self.logger.info("RAG agent initialized with simple TF-IDF embeddings")
                        
                    except Exception as tfidf_e:
                        self.logger.error(f"Even simple TF-IDF embeddings failed: {tfidf_e}")
                        raise ValueError("All embedding model initialization attempts failed")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG agent: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            self.logger.info(f"==== RAG AGENT INITIALIZATION END ====")
    
    def create_collection_name(self, filepath: str) -> str:
        """Create a valid collection name from a filepath
        
        Args:
            filepath: Path to the file
            
        Returns:
            A valid collection name
        """
        # Extract filename without extension
        collection_name = Path(filepath).stem
        
        # Fix potential issues from naming convention
        collection_name = collection_name.replace(" ", "-")
        collection_name = unidecode(collection_name)
        collection_name = re.sub("[^A-Za-z0-9]+", "-", collection_name)
        collection_name = collection_name[:50]
        
        if len(collection_name) < 3:
            collection_name = collection_name + "xyz"
        if not collection_name[0].isalnum():
            collection_name = "A" + collection_name[1:]
        if not collection_name[-1].isalnum():
            collection_name = collection_name[:-1] + "Z"
            
        self.logger.info(f"Collection name: {collection_name}")
        return collection_name
    
    def get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding with caching to avoid recomputing
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding
        """
        # Use hash as key to handle longer texts
        text_hash = hash(text)
        
        # Check if in cache
        cached_embedding = self.embedding_cache.get(text_hash)
        if cached_embedding is not None:
            return cached_embedding
        
        # Wait if resources are constrained
        while self.resource_monitor.paused.is_set():
            time.sleep(0.1)
            if self.cancellation_token.is_set():
                raise InterruptedError("Operation was cancelled")
        
        embedding = self.embedding_model.embed_query(text)
        self.embedding_cache.put(text_hash, embedding)
        return embedding
    
    def create_dynamic_chunks(self, text: str) -> List[str]:
        """Use paragraph and section boundaries to create more semantic chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_text(text)
    
    def load_document(self, file_path: str, callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """Load a document, process it, create vector database, and handle empty/unreadable PDFs.

        Args:
            file_path: Path to the document
            callback: Optional callback for progress updates (message, progress percentage)

        Returns:
            Document information dictionary or error dictionary
        """
        # Reset cancellation token
        self.cancellation_token.clear()
        
        try:
            self.logger.info(f"Starting document processing for: {file_path}")
            if callback:
                callback("Starting document processing...", 0.0)

            # --- Step 1: Load and Split Document ---
            doc_splits = self.load_doc(file_path, callback)
            
            # Check for cancellation
            if self.cancellation_token.is_set():
                return {
                    "success": False,
                    "error": "Operation was cancelled by the user"
                }

            # --- Step 2: Validate Extracted Content ---
            if not doc_splits or not any(chunk.page_content.strip() for chunk in doc_splits):
                error_message = (
                    "Could not extract meaningful text from the PDF. "
                    "It might be an image-only PDF, corrupted, password-protected, or empty. "
                    "Please provide a document with selectable text."
                )
                self.logger.error(f"{error_message} File: {file_path}")
                return {
                    "success": False,
                    "error": error_message
                }

            # Log successful splitting
            self.logger.info(f"Successfully extracted {len(doc_splits)} text chunks from {file_path}")
            if callback:
                callback(f"Successfully extracted {len(doc_splits)} text chunks", 30.0)

            # --- Step 3: Create Collection Name ---
            collection_name = self.create_collection_name(file_path)
            self.logger.info(f"Generated collection name: {collection_name}")

            # --- Step 4: Create Vector Database ---
            try:
                self.logger.info(f"Creating FAISS vector database for {collection_name}...")
                if callback:
                    callback("Creating vector database...", 40.0)
                
                vector_db = self.create_optimized_db(doc_splits, collection_name, callback)
                
                # Check for cancellation
                if self.cancellation_token.is_set():
                    return {
                        "success": False,
                        "error": "Operation was cancelled by the user"
                    }
                
                self.logger.info(f"Vector database created successfully for {collection_name}.")

                # Create TF-IDF vectorizer if using hybrid search
                if self.use_hybrid_search:
                    try:
                        self.logger.info(f"Creating TF-IDF vectorizer for {collection_name}...")
                        if callback:
                            callback("Creating keyword search index...", 80.0)
                        
                        tfidf_vectorizer = TfidfVectorizer()
                        tfidf_vectorizer.fit([doc.page_content for doc in doc_splits])
                        self.tfidf_vectorizers[collection_name] = tfidf_vectorizer
                        self.logger.info(f"TF-IDF vectorizer created for {collection_name}.")
                    except Exception as tfidf_error:
                        self.logger.warning(f"Could not create TF-IDF vectorizer for {collection_name}: {tfidf_error}. Hybrid search may be affected.")
                        self.tfidf_vectorizers.pop(collection_name, None)

            except Exception as db_error:
                self.logger.error(f"Error creating vector database for {collection_name}: {db_error}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {
                    "success": False,
                    "error": f"Failed to create vector database: {db_error}"
                }

            # --- Step 5: Store Document Info and Return Success ---
            chunk_metadata = [{
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", file_path),
                "content_length": len(doc.page_content)
            } for doc in doc_splits]
            
            document_info = {
                "id": collection_name,
                "path": file_path,
                "name": os.path.basename(file_path),
                "chunk_metadata": chunk_metadata,
                "vector_db": vector_db,
                "total_chunks": len(doc_splits)
            }

            # Store in the agent's documents dictionary
            self.documents[collection_name] = document_info
            
            # Force garbage collection to clean up memory after processing
            gc.collect()
            
            if callback:
                callback("Document processing complete", 100.0)

            success_message = f"Document '{os.path.basename(file_path)}' uploaded and processed successfully with {len(doc_splits)} chunks."
            self.logger.info(success_message)
            return {
                "document_id": collection_name,
                "file_name": os.path.basename(file_path),
                "chunks": len(doc_splits),
                "success": True,
                "message": success_message
            }

        except Exception as e:
            # Catch-all for any unexpected errors during the process
            self.logger.error(f"Unexpected error during document processing for {file_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"An unexpected error occurred: {e}"
            }
    
    def process_page(self, page: Document) -> List[Document]:
        """Process a single page into chunks
        
        Args:
            page: Document page
            
        Returns:
            List of document chunks
        """
        # Check for cancellation
        if self.cancellation_token.is_set():
            return []
            
        # Wait if resources are constrained
        while self.resource_monitor.paused.is_set():
            time.sleep(0.1)
            if self.cancellation_token.is_set():
                return []
        
        if self.use_dynamic_chunking:
            # Apply dynamic chunking
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
        else:
            # Use standard chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
        
        return text_splitter.split_documents([page])
    
    def load_doc(self, file_path: str, callback: Optional[Callable[[str, float], None]] = None) -> List[Document]:
        """Load PDF document with optimized chunking and parallel processing
        
        Args:
            file_path: Path to the PDF document
            callback: Optional callback for progress updates
            
        Returns:
            List of document chunks
        """
        self.logger.info(f"Loading document: {file_path}")
        if callback:
            callback("Loading document...", 5.0)
        
        # Load document
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        if callback:
            callback(f"Loaded {len(pages)} pages", 10.0)
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid window spawning
        if self.use_parallel and len(pages) >= 5:  # Only use parallel for larger documents
            self.logger.info(f"Parallel processing for document with {len(pages)} pages")
            if callback:
                callback("Processing pages in parallel...", 15.0)
            
            chunks = []
            processed_pages = 0
            
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Process pages in parallel
                futures = [executor.submit(self.process_page, page) for page in pages]
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    if self.cancellation_token.is_set():
                        return []
                    
                    try:
                        page_chunks = future.result()
                        chunks.extend(page_chunks)
                        processed_pages += 1
                        
                        # Update progress
                        if callback:
                            progress = 15.0 + (processed_pages / len(pages)) * 15.0
                            callback(f"Processed {processed_pages}/{len(pages)} pages", progress)
                    except Exception as e:
                        self.logger.error(f"Error processing page: {str(e)}")
            
            doc_splits = chunks
        else:
            # Standard sequential processing for smaller documents
            self.logger.info(f"Standard processing for document with {len(pages)} pages")
            if callback:
                callback("Processing document...", 15.0)
            
            if self.use_dynamic_chunking:
                # Apply dynamic chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ". ", " ", ""],
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                )
            else:
                # Use standard chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, 
                    chunk_overlap=self.chunk_overlap
                )
            
            doc_splits = text_splitter.split_documents(pages)
            if callback:
                callback("Document processing complete", 25.0)
        
        self.logger.info(f"Document split into {len(doc_splits)} chunks")
        return doc_splits
    
    def create_optimized_db(self, splits: List[Document], collection_name: str, 
                           callback: Optional[Callable[[str, float], None]] = None) -> FAISS:
        """Create optimized FAISS index for better performance
        
        Args:
            splits: List of document chunks
            collection_name: Name for the FAISS index
            callback: Optional callback for progress updates
            
        Returns:
            FAISS vector store
        """
        self.logger.info("Creating optimized FAISS vector database...")
        
        # Initialize the embedding model if not already initialized
        if not hasattr(self, 'embedding_model') or self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
        
        # Create FAISS index from documents - use standard method for small datasets
        if len(splits) < 100:
            self.logger.info(f"Using standard FAISS index for {len(splits)} chunks")
            if callback:
                callback("Creating vector database...", 50.0)
            
            # Wait if resources are constrained
            while self.resource_monitor.paused.is_set():
                time.sleep(0.1)
                if self.cancellation_token.is_set():
                    raise InterruptedError("Operation was cancelled")
            
            vectordb = FAISS.from_documents(
                documents=splits,
                embedding=self.embedding_model
            )
        else:
            # Get document embeddings - use batched processing for larger datasets
            self.logger.info(f"Using optimized IVF FAISS index for {len(splits)} chunks")
            
            # Process in smaller batches to manage memory
            batch_size = 50  # Adjust based on memory constraints
            total_batches = (len(splits) + batch_size - 1) // batch_size
            
            all_embeddings = []
            for i in range(0, len(splits), batch_size):
                if self.cancellation_token.is_set():
                    raise InterruptedError("Operation was cancelled")
                
                # Wait if resources are constrained
                while self.resource_monitor.paused.is_set():
                    time.sleep(0.1)
                    if self.cancellation_token.is_set():
                        raise InterruptedError("Operation was cancelled")
                
                batch = splits[i:i+batch_size]
                batch_texts = [doc.page_content for doc in batch]
                
                if callback:
                    progress = 40.0 + ((i // batch_size) / total_batches) * 30.0
                    callback(f"Creating embeddings (batch {i // batch_size + 1}/{total_batches})...", progress)
                
                batch_embeddings = self.embedding_model.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Force garbage collection between batches to manage memory
                if i % (batch_size * 3) == 0 and i > 0:
                    gc.collect()
            
            # Convert to numpy array
            # Convert to numpy array
            embeddings_np = np.array(all_embeddings).astype('float32')
            
            # Create optimized FAISS index with IVF for faster search
            dimension = embeddings_np.shape[1]
            nlist = min(int(np.sqrt(len(splits))), 100)  # Rule of thumb: nlist ~= sqrt(n)
            
            if callback:
                callback("Building search index...", 70.0)
            
            # Create and train the index
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Need to train index
            if len(splits) >= nlist:
                # Wait if resources are constrained
                while self.resource_monitor.paused.is_set():
                    time.sleep(0.1)
                    if self.cancellation_token.is_set():
                        raise InterruptedError("Operation was cancelled")
                        
                index.train(embeddings_np)
                index.add(embeddings_np)
            else:
                # Fall back to simple index if too few samples
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings_np)
            
            # Create FAISS vector store
            vectordb = FAISS(
                embedding_function=self.embedding_model.embed_query,
                index=index,
                docstore=InMemoryDocstore(dict(zip(range(len(splits)), splits))),
                index_to_docstore_id=dict(zip(range(len(splits)), range(len(splits))))
            )
        
        # Save the FAISS index to disk
        index_path = os.path.join(self.faiss_index_dir, collection_name)
        vectordb.save_local(index_path)
        self.logger.info(f"FAISS vector database created and saved to {index_path}")
        
        if callback:
            callback("Vector database created and saved", 75.0)
        
        return vectordb
    
    def hybrid_search(self, query: str, document_id: str, k: int = 3) -> List[Document]:
        """Combine keyword and semantic search for better results
        
        Args:
            query: Query string
            document_id: Document ID
            k: Number of results to return
            
        Returns:
            List of most relevant document chunks
        """
        if document_id not in self.documents:
            raise ValueError(f"Document ID not found: {document_id}")
            
        # Get vector db
        vector_db = self.documents[document_id]["vector_db"]
        
        # Get TF-IDF vectorizer or create one if it doesn't exist
        if document_id not in self.tfidf_vectorizers:
            self.logger.info(f"TF-IDF vectorizer not found for document {document_id}")
            # Hybrid search won't be effective - fall back to semantic search
            return vector_db.similarity_search(query, k=k)
        
        # Get chunk content from vector store for TF-IDF search
        tfidf_vectorizer = self.tfidf_vectorizers[document_id]
        
        # Get all documents from the vector store's docstore
        all_docs = list(vector_db.docstore._dict.values())
        
        # Keyword search with TF-IDF
        try:
            tfidf_matrix = tfidf_vectorizer.transform([doc.page_content for doc in all_docs])
            query_vec = tfidf_vectorizer.transform([query])
            keyword_scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
        except Exception as e:
            self.logger.error(f"Error in TF-IDF search: {str(e)}")
            # Fall back to semantic search if TF-IDF fails
            return vector_db.similarity_search(query, k=k)
        
        # Semantic search with embeddings
        semantic_results = vector_db.similarity_search_with_score(query, k=k)
        
        # Map semantic results to indices
        semantic_indices = []
        semantic_scores = []
        for doc, score in semantic_results:
            # Find index of the document in all_docs
            for i, chunk in enumerate(all_docs):
                if chunk.page_content == doc.page_content:
                    semantic_indices.append(i)
                    # Convert distance to similarity (1.0 - normalized distance)
                    semantic_scores.append(1.0 - min(score / 10.0, 1.0))
                    break
        
        # Combine scores (weighted sum of semantic and keyword)
        combined_scores = np.zeros(len(all_docs))
        
        # Set semantic scores
        for idx, score in zip(semantic_indices, semantic_scores):
            combined_scores[idx] += self.semantic_weight * score
        
        # Add keyword scores (normalized)
        if np.max(keyword_scores) > 0:
            normalized_keyword_scores = keyword_scores / np.max(keyword_scores)
            combined_scores += self.keyword_weight * normalized_keyword_scores
        
        # Get top k results
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        
        return [all_docs[i] for i in top_indices]
    
    def setup_retriever(self, vector_db):
        """Set up retriever with MMR for more diverse results
        
        Args:
            vector_db: Vector database
            
        Returns:
            Configured retriever
        """
        if self.use_mmr:
            # Use Maximum Marginal Relevance
            return vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': self.top_k,
                    'fetch_k': self.top_k * 3,  # Consider more documents for diversity
                    'lambda_mult': self.mmr_lambda  # Balance between relevance and diversity
                }
            )
        else:
            # Use standard similarity search
            return vector_db.as_retriever(
                search_kwargs={'k': self.top_k}
            )
    
    def cleanup_llm_resources(self):
        """Clean up LLM resources properly before switching models"""
        # Clean up previous LLM if it exists
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # Delete the LLM to free resources
                del self.llm
                self.llm = None
                
                # Force garbage collection
                gc.collect()
                
                self.logger.info("Successfully cleaned up previous LLM resources")
            except Exception as e:
                self.logger.error(f"Error cleaning up LLM resources: {str(e)}")
    
    def initialize_llmchain(self, document_id: str) -> bool:
        """Initialize the LLM chain for a specific document using API providers
        
        Args:
            document_id: Document ID to use
            
        Returns:
            True if successful, False otherwise
        """
        # Reset cancellation token
        self.cancellation_token.clear()
        
        try:
            # Check if document exists
            if document_id not in self.documents:
                self.logger.error(f"Document ID not found: {document_id}")
                return False
            
            # Get vector database for document
            vector_db = self.documents[document_id]["vector_db"]
            
            # Clean up previous resources if model is changing
            if self.llm is not None:
                self.cleanup_llm_resources()
            
            self.logger.info(f"Initializing LLM chain with API provider: {self.api_provider}")
            
            # Use API providers instead of HuggingFace
            from app.utils.api_providers import APIProviderFactory
            from app.core.config_manager import ConfigManager
            from langchain.llms.base import LLM
            
            config_manager = ConfigManager()
            api_keys = {
                "openai": config_manager.get_openai_api_key(),
                "gemini": config_manager.get_gemini_api_key(),
                "groq": config_manager.get_groq_api_key(),
                "anthropic": config_manager.get_anthropic_api_key()
            }
            
            api_key = api_keys.get(self.api_provider)
            if not api_key:
                self.logger.error(f"{self.api_provider.upper()} API key required")
                return False
            
            # Create custom LLM wrapper for langchain
            class CustomAPILLM(LLM):
                def __init__(self, provider, api_key, model_id, temperature, max_tokens):
                    super().__init__()
                    self.provider_instance = APIProviderFactory.create_provider(provider, api_key, model_id)
                    self.temperature = temperature
                    self.max_tokens = max_tokens
                
                def _call(self, prompt, stop=None):
                    messages = [{"content": prompt}]
                    return self.provider_instance.generate(
                        messages, 
                        temperature=self.temperature, 
                        max_tokens=self.max_tokens
                    )
                
                @property
                def _llm_type(self):
                    return f"custom_{self.api_provider}"
            
            self.llm = CustomAPILLM(self.api_provider, api_key, self.model_id, self.temperature, self.max_tokens)
            self.logger.info(f"Initialized {self.api_provider.upper()} LLM successfully")
            
            # For RAG, we'll implement a simpler approach without conversational memory
            # since we're not using HuggingFace endpoints anymore
            
            self.logger.info("LLM chain initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM chain: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Clean up any partial resources
            self.cleanup_llm_resources()
            return False
    
    def format_chat_history(self, chat_history):
        """Format chat history for llm chain
        
        Args:
            chat_history: List of (user, bot) message pairs
            
        Returns:
            Formatted chat history
        """
        formatted_chat_history = []
        for user_message, bot_message in chat_history:
            formatted_chat_history.append(f"User: {user_message}")
            formatted_chat_history.append(f"Assistant: {bot_message}")
        return formatted_chat_history
    
    def cancel_current_operation(self):
        """Cancel the current operation"""
        self.cancellation_token.set()
        self.logger.info("Operation cancellation requested")
        return {"success": True, "message": "Operation cancelled"}
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input
        
        Args:
            input_text: Input text for the agent
            callback: Optional callback for streaming responses
            
        Returns:
            Agent output text
        """
        # Reset cancellation token at the start of each run
        self.cancellation_token.clear()
        
        def progress_callback(message, progress=None):
            """Callback for progress updates that adds percentage if available"""
            if progress is not None:
                callback(f"{message} ({progress:.0f}%)")
            else:
                callback(message)
        
        try:
            # Check if this is a command in JSON format
            try:
                command = json.loads(input_text)
                action = command.get("action", "")
                
                if action == "cancel":
                    # Cancel current operation
                    return json.dumps(self.cancel_current_operation())
                
                elif action == "upload":
                    # Upload a document
                    file_path = command.get("file_path", "")
                    if not file_path:
                        return json.dumps({
                            "success": False,
                            "error": "No file path provided"
                        })
                    
                    if callback:
                        callback("Uploading and processing document...")
                    
                    # Use a wrapper callback that includes progress percentage
                    cb = progress_callback if callback else None
                    
                    # Load the document
                    result = self.load_document(file_path, cb)
                    
                    # Return JSON result
                    return json.dumps(result)
                    
                elif action == "query":
                    # Query a document
                    document_id = command.get("document_id", "")
                    query = command.get("query", "")
                    
                    if not document_id:
                        return json.dumps({
                            "success": False,
                            "error": "No document ID provided"
                        })
                    
                    if not query:
                        return json.dumps({
                            "success": False,
                            "error": "No query provided"
                        })
                    
                    if callback:
                        callback("Processing query...")
                    
                    # Initialize LLM if needed
                    if self.llm is None or document_id != self.current_document_id:
                        if callback:
                            callback("Initializing language model...")
                        
                        self.current_document_id = document_id
                        success = self.initialize_llmchain(document_id)
                        
                        if not success:
                            return json.dumps({
                                "success": False,
                                "error": f"Failed to initialize LLM chain for document {document_id}"
                            })
                    
                    # Use hybrid search if enabled
                    if self.use_hybrid_search and query:
                        self.logger.info(f"Using hybrid search for query: {query}")
                        
                        if callback:
                            callback("Performing hybrid search...")
                        
                        # Perform hybrid search
                        try:
                            retrieved_docs = self.hybrid_search(query, document_id, k=self.top_k)
                            
                            if callback:
                                callback("Generating response...")
                            
                            # Check for cancellation
                            if self.cancellation_token.is_set():
                                return json.dumps({
                                    "success": False,
                                    "error": "Operation was cancelled by the user"
                                })
                            
                            # Generate response with LLM using retrieved documents as context
                            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            
                            prompt = self.SYSTEM_PROMPT.format(
                                question=query,
                                context=context
                            )
                            
                            # Wait if resources are constrained
                            while self.resource_monitor.paused.is_set():
                                time.sleep(0.1)
                                if self.cancellation_token.is_set():
                                    return json.dumps({
                                        "success": False,
                                        "error": "Operation was cancelled by the user"
                                    })
                            
                            # Use the API provider directly
                            messages = [{"content": prompt}]
                            answer = self.llm.generate(
                                messages, 
                                temperature=self.temperature, 
                                max_tokens=self.max_tokens
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Error in hybrid search: {str(e)}")
                            # Fall back to standard retrieval
                            if callback:
                                callback("Falling back to standard retrieval...")
                            
                            # Wait if resources are constrained
                            while self.resource_monitor.paused.is_set():
                                time.sleep(0.1)
                                if self.cancellation_token.is_set():
                                    return json.dumps({
                                        "success": False,
                                        "error": "Operation was cancelled by the user"
                                    })
                            
                            # Use basic similarity search
                            vector_db = self.documents[document_id]["vector_db"]
                            retrieved_docs = vector_db.similarity_search(query, k=self.top_k)
                            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            
                            prompt = self.SYSTEM_PROMPT.format(
                                question=query,
                                context=context
                            )
                            
                            messages = [{"content": prompt}]
                            answer = self.llm.generate(
                                messages, 
                                temperature=self.temperature, 
                                max_tokens=self.max_tokens
                            )
                    else:
                        # Standard retrieval
                        if callback:
                            callback("Retrieving information...")
                        
                        # Wait if resources are constrained
                        while self.resource_monitor.paused.is_set():
                            time.sleep(0.1)
                            if self.cancellation_token.is_set():
                                return json.dumps({
                                    "success": False,
                                    "error": "Operation was cancelled by the user"
                                })
                        
                        # Use basic similarity search
                        vector_db = self.documents[document_id]["vector_db"]
                        retrieved_docs = vector_db.similarity_search(query, k=self.top_k)
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        
                        prompt = self.SYSTEM_PROMPT.format(
                            question=query,
                            context=context
                        )
                        
                        messages = [{"content": prompt}]
                        answer = self.llm.generate(
                            messages, 
                            temperature=self.temperature, 
                            max_tokens=self.max_tokens
                        )
                    
                    # Clean up the answer
                    if answer.find("Helpful Answer:") != -1:
                        answer = answer.split("Helpful Answer:")[-1]
                    
                    # Format sources
                    sources = []
                    for i, doc in enumerate(retrieved_docs[:self.top_k]):
                        page_num = doc.metadata.get("page", 0) + 1
                        content = doc.page_content.strip()
                        sources.append({
                            "page": page_num,
                            "content": content[:300] + ("..." if len(content) > 300 else "")
                        })
                    
                    # Add to history
                    self.add_to_history(query, answer)
                    
                    if callback:
                        callback("Query processing complete")
                    
                    # Return JSON result
                    return json.dumps({
                        "success": True,
                        "answer": answer,
                        "sources": sources
                    })
                
                # Add custom action for updating agent configuration
                elif action == "update_config":
                    config_updates = command.get("config", {})
                    
                    # Validate configuration updates
                    if "model_id" in config_updates:
                        new_model = config_updates["model_id"]
                        self.logger.info(f"Updating model to: {new_model}")
                        
                        # Clean up previous resources before changing model
                        self.cleanup_llm_resources()
                    
                    # Update configuration parameters
                    for key, value in config_updates.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                            self.logger.info(f"Updated config parameter: {key} = {value}")
                    
                    # Reset the LLM to apply new configuration
                    self.llm = None
                    self.current_document_id = None
                    
                    # Force garbage collection
                    gc.collect()
                    
                    return json.dumps({
                        "success": True,
                        "message": "Configuration updated successfully"
                    })
                
                else:
                    return json.dumps({
                        "success": False,
                        "error": f"Unknown action: {action}"
                    })
                    
            except json.JSONDecodeError:
                # Not a JSON command, treat as a regular query
                if callback:
                    callback("Please use the query function to ask questions about documents")
                
                return "Please upload a document first and then use the query function."
                
        except Exception as e:
            self.logger.error(f"Error in RAG agent: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Try to clean up resources on error
            try:
                self.cleanup_llm_resources()
                gc.collect()
            except:
                pass
                
            return json.dumps({
                "success": False,
                "error": f"Error: {str(e)}"
            })
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
        self.cleanup_llm_resources()
        self.current_document_id = None
        
        # Clear caches to free memory
        self.embedding_cache.clear()
        
        # Force garbage collection
        gc.collect()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        capabilities = [
            "document_retrieval",
            "question_answering", 
            "pdf_processing",
            "api_based_llm"
        ]
        
        if self.use_hybrid_search:
            capabilities.append("hybrid_search")
        
        if self.use_dynamic_chunking:
            capabilities.append("dynamic_chunking")
            
        if self.use_mmr:
            capabilities.append("diverse_retrieval")
            
        return capabilities