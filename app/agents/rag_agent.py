"""
Simplified RAG Agent for sagax1
Clean implementation with API-first approach for both embeddings and chat
"""

import os
import logging
import json
import uuid
import pickle
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import tempfile

from app.agents.base_agent import BaseAgent


class SimpleDocument:
    """Simple document class to replace langchain Document"""
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}


class SimpleEmbeddings:
    """Simple embeddings class that can use either API or sentence-transformers"""
    
    def __init__(self, provider: str = "openai", api_key: str = None, model: str = None):
        self.provider = provider
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Set default models
        if not model:
            if provider == "openai":
                self.model = "text-embedding-3-small"
            elif provider == "sentence-transformers":
                self.model = "all-MiniLM-L6-v2"
            else:
                self.model = "text-embedding-3-small"  # Default fallback
        else:
            self.model = model
            
        # Initialize sentence transformers if using local embeddings
        if provider == "sentence-transformers":
            self._init_sentence_transformers()
    
    def _init_sentence_transformers(self):
        """Initialize sentence transformers for local embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model)
            self.logger.info(f"Loaded sentence transformer model: {self.model}")
        except ImportError:
            self.logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            self.logger.error(f"Error loading sentence transformer: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        if self.provider == "sentence-transformers":
            return self._embed_with_sentence_transformers([text])[0]
        else:
            return self._embed_with_api([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        if self.provider == "sentence-transformers":
            return self._embed_with_sentence_transformers(texts)
        else:
            return self._embed_with_api(texts)
    
    def _embed_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using sentence transformers"""
        try:
            embeddings = self.st_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error creating embeddings with sentence transformers: {e}")
            raise
    
    def _embed_with_api(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using API providers"""
        if self.provider == "openai":
            return self._embed_with_openai(texts)
        else:
            # For other providers that don't have embedding APIs, fall back to sentence transformers
            self.logger.warning(f"Provider {self.provider} doesn't support embeddings, falling back to sentence transformers")
            if not hasattr(self, 'st_model'):
                try:
                    self._init_sentence_transformers()
                except Exception as e:
                    self.logger.error(f"Failed to initialize sentence transformers: {e}")
                    raise Exception("Cannot create embeddings: OpenAI API unavailable and sentence-transformers failed to load")
            return self._embed_with_sentence_transformers(texts)
    
    def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # OpenAI has a limit on batch size, so process in chunks
            all_embeddings = []
            batch_size = 100  # OpenAI's limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            self.logger.error(f"Error creating embeddings with OpenAI: {e}")
            raise


class SimpleVectorStore:
    """Simple vector store using numpy for similarity search"""
    
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.index_to_doc = {}
    
    def add_documents(self, documents: List[SimpleDocument], embeddings: List[List[float]]):
        """Add documents and their embeddings to the store"""
        start_idx = len(self.documents)
        
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        
        # Update index mapping
        for i, doc in enumerate(documents):
            self.index_to_doc[start_idx + i] = doc
    
    def similarity_search(self, query_embedding: List[float], k: int = 3) -> List[SimpleDocument]:
        """Search for similar documents"""
        if not self.embeddings:
            return []
        
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        embeddings_matrix = np.array(self.embeddings)
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings_matrix, query_vec) / (
            np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return corresponding documents
        return [self.index_to_doc[i] for i in top_indices if i in self.index_to_doc]
    
    def save(self, filepath: str):
        """Save the vector store to disk"""
        data = {
            'embeddings': self.embeddings,
            'documents': [(doc.content, doc.metadata) for doc in self.documents],
            'index_to_doc': {k: (v.content, v.metadata) for k, v in self.index_to_doc.items()}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load the vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.documents = [SimpleDocument(content, metadata) for content, metadata in data['documents']]
        self.index_to_doc = {k: SimpleDocument(content, metadata) for k, (content, metadata) in data['index_to_doc'].items()}


class SimplifiedRAGAgent(BaseAgent):
    """Simplified RAG Agent with API-first approach"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the simplified RAG agent"""
        super().__init__(agent_id, config)
        
        # Configuration
        self.api_provider = config.get("api_provider", "openai")
        self.model_id = config.get("model_id", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.top_k = config.get("top_k", 3)
        
        # Embedding configuration - Default to sentence-transformers for most providers
        self.embedding_provider = config.get("embedding_provider", "sentence-transformers")
        if self.api_provider == "openai" and self.embedding_provider not in ["sentence-transformers"]:
            # Only use OpenAI embeddings if explicitly using OpenAI and not overridden
            self.embedding_provider = "openai"
        elif self.api_provider in ["groq", "gemini", "anthropic"]:
            # Force sentence-transformers for providers that don't have embedding APIs
            self.embedding_provider = "sentence-transformers"
        self.embedding_model = config.get("embedding_model", None)
        
        # Storage
        self.documents_dir = config.get("documents_dir", "./rag_documents")
        os.makedirs(self.documents_dir, exist_ok=True)
        
        # Initialize components
        self.embeddings = None
        self.vector_stores = {}  # document_id -> SimpleVectorStore
        self.api_provider_instance = None
        
        self.logger.info(f"Simplified RAG Agent initialized with {self.api_provider} for chat and {self.embedding_provider} for embeddings")
    
    def _get_api_key(self) -> str:
        """Get API key for the current provider"""
        from app.core.config_manager import ConfigManager
        config_manager = ConfigManager()
        
        if self.api_provider == "openai":
            return config_manager.get_openai_api_key()
        elif self.api_provider == "groq":
            return config_manager.get_groq_api_key()
        elif self.api_provider == "gemini":
            return config_manager.get_gemini_api_key()
        elif self.api_provider == "anthropic":
            return config_manager.get_anthropic_api_key()
        else:
            return None
    
    def _initialize_embeddings(self):
        """Initialize the embeddings model"""
        if self.embeddings is not None:
            return
        
        # Default to sentence-transformers if no specific embedding provider is set
        # or if the API provider doesn't support embeddings
        if self.embedding_provider == "sentence-transformers" or self.api_provider in ["groq", "gemini", "anthropic"]:
            self.embedding_provider = "sentence-transformers"
            api_key = None
            self.logger.info(f"Using sentence-transformers for embeddings (chat provider: {self.api_provider})")
        else:
            # Only use API embeddings if explicitly set to openai
            api_key = None
            if self.embedding_provider == "openai":
                from app.core.config_manager import ConfigManager
                config_manager = ConfigManager()
                api_key = config_manager.get_openai_api_key()
                
                if not api_key:
                    self.logger.warning("No OpenAI API key found, falling back to sentence-transformers")
                    self.embedding_provider = "sentence-transformers"
        
        self.embeddings = SimpleEmbeddings(
            provider=self.embedding_provider,
            api_key=api_key,
            model=self.embedding_model
        )
        self.logger.info(f"Initialized embeddings with provider: {self.embedding_provider}")
    
    def _initialize_chat_api(self):
        """Initialize the chat API provider"""
        if self.api_provider_instance is not None:
            return
        
        from app.utils.api_providers import APIProviderFactory
        
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(f"No API key found for {self.api_provider}")
        
        self.api_provider_instance = APIProviderFactory.create_provider(
            self.api_provider, api_key, self.model_id
        )
        self.logger.info(f"Initialized chat API: {self.api_provider}")
    
    def _split_text(self, text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
        """Simple text splitting"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:  # Don't make chunks too small
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _load_pdf(self, file_path: str) -> str:
        """Load text from PDF file"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            self.logger.error("PyPDF2 not available. Install with: pip install PyPDF2")
            raise
        except Exception as e:
            self.logger.error(f"Error loading PDF: {e}")
            raise
    
    def load_document(self, file_path: str, callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """Load and process a document"""
        try:
            self._initialize_embeddings()
            
            if callback:
                callback("Loading document...", 10.0)
            
            # Load text based on file type
            if file_path.lower().endswith('.pdf'):
                text = self._load_pdf(file_path)
            elif file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                return {
                    "success": False,
                    "error": "Unsupported file type. Only PDF and TXT files are supported."
                }
            
            if not text.strip():
                return {
                    "success": False,
                    "error": "No text could be extracted from the document."
                }
            
            if callback:
                callback("Splitting text into chunks...", 30.0)
            
            # Split text into chunks
            chunks = self._split_text(text)
            self.logger.info(f"Split document into {len(chunks)} chunks")
            
            if callback:
                callback("Creating embeddings...", 50.0)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = SimpleDocument(
                    content=chunk,
                    metadata={"source": file_path, "chunk_id": i, "page": i}
                )
                documents.append(doc)
            
            # Create embeddings
            embeddings = self.embeddings.embed_texts([doc.content for doc in documents])
            
            if callback:
                callback("Building vector store...", 80.0)
            
            # Create vector store
            vector_store = SimpleVectorStore()
            vector_store.add_documents(documents, embeddings)
            
            # Generate document ID
            document_id = Path(file_path).stem + "_" + str(uuid.uuid4().hex[:8])
            
            # Save vector store
            store_path = os.path.join(self.documents_dir, f"{document_id}.pkl")
            vector_store.save(store_path)
            
            # Store in memory
            self.vector_stores[document_id] = vector_store
            
            if callback:
                callback("Document processing complete", 100.0)
            
            return {
                "success": True,
                "document_id": document_id,
                "file_name": os.path.basename(file_path),
                "chunks": len(chunks),
                "message": f"Document processed successfully with {len(chunks)} chunks"
            }
        
        except Exception as e:
            self.logger.error(f"Error loading document: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def query_document(self, document_id: str, query: str) -> Dict[str, Any]:
        """Query a document"""
        try:
            self._initialize_embeddings()
            self._initialize_chat_api()
            
            # Load vector store if not in memory
            if document_id not in self.vector_stores:
                store_path = os.path.join(self.documents_dir, f"{document_id}.pkl")
                if not os.path.exists(store_path):
                    return {
                        "success": False,
                        "error": f"Document {document_id} not found"
                    }
                
                vector_store = SimpleVectorStore()
                vector_store.load(store_path)
                self.vector_stores[document_id] = vector_store
            
            vector_store = self.vector_stores[document_id]
            
            # Create query embedding
            query_embedding = self.embeddings.embed_text(query)
            
            # Search for relevant documents
            relevant_docs = vector_store.similarity_search(query_embedding, k=self.top_k)
            
            if not relevant_docs:
                return {
                    "success": False,
                    "error": "No relevant documents found"
                }
            
            # Create context from relevant documents
            context = "\n\n".join([doc.content for doc in relevant_docs])
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            messages = [{"content": prompt}]
            answer = self.api_provider_instance.generate(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Format sources
            sources = []
            for doc in relevant_docs:
                sources.append({
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata
                })
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources
            }
        
        except Exception as e:
            self.logger.error(f"Error querying document: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input"""
        try:
            # Parse input as JSON command
            command = json.loads(input_text)
            action = command.get("action", "")
            
            if action == "upload":
                file_path = command.get("file_path", "")
                if not file_path:
                    return json.dumps({
                        "success": False,
                        "error": "No file path provided"
                    })
                
                def progress_callback(message, progress):
                    if callback:
                        callback(f"{message} ({progress:.0f}%)")
                
                result = self.load_document(file_path, progress_callback)
                return json.dumps(result)
            
            elif action == "query":
                document_id = command.get("document_id", "")
                query = command.get("query", "")
                
                if not document_id or not query:
                    return json.dumps({
                        "success": False,
                        "error": "Document ID and query are required"
                    })
                
                if callback:
                    callback("Processing query...")
                
                result = self.query_document(document_id, query)
                return json.dumps(result)
            
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown action: {action}"
                })
        
        except json.JSONDecodeError:
            return "Please provide commands in JSON format with 'action' field."
        except Exception as e:
            self.logger.error(f"Error in RAG agent: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
        self.vector_stores.clear()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has"""
        return [
            "document_retrieval",
            "question_answering",
            "pdf_processing",
            "txt_processing",
            f"{self.api_provider}_chat",
            f"{self.embedding_provider}_embeddings"
        ]