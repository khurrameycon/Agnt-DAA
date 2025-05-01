"""
RAG Agent for sagax1
Agent for document retrieval and question answering using FAISS vector store
Optimized for performance with dynamic chunking, caching, and hybrid search
"""

import os
import logging
import tempfile
import uuid
import json
import re
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union

from app.agents.base_agent import BaseAgent

# Import necessary libraries for document processing
from unidecode import unidecode
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import necessary libraries for vector database
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# Import necessary libraries for LLM
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from app.core.config_manager import ConfigManager
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
        self.model_id = config.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.top_k = config.get("top_k", 3)
        
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
        
        # Parallel processing
        self.use_parallel = config.get("use_parallel", True)
        self.max_workers = config.get("max_workers", min(8, os.cpu_count() or 1))
        # Replacement logic:
        self.api_key = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_API_KEY")
        if not self.api_key:
            try:
                # Use ConfigManager to get the key
                config_manager = ConfigManager() # Instantiate ConfigManager
                self.api_key = config_manager.get_hf_api_key()
                if self.api_key:
                    self.logger.info("API key successfully retrieved from ConfigManager.")
                else:
                    self.logger.warning("API key not found in environment variables or config file.")
            except Exception as e:
                self.logger.error(f"Error retrieving API key using ConfigManager: {str(e)}")
                self.api_key = None # Ensure api_key is None if retrieval failed

        # Make sure the HuggingFaceEndpoint uses this self.api_key
        # (Check the initialize_llmchain method)
        # Set base directory for storing FAISS indexes
        self.faiss_index_dir = config.get("faiss_index_dir", "./faiss_indexes")
        os.makedirs(self.faiss_index_dir, exist_ok=True)
        
        # Status and components
        self.qa_chain = None
        self.documents = {}  # Track loaded documents
        self.current_document_id = None
        self.is_initialized = False
        
        # Caching
        self.embedding_cache = {}
        self.tfidf_vectorizers = {}
        
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
    
    def initialize(self) -> None:
        """Initialize embedding model"""
        if self.is_initialized:
            return
        
        try:
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            )
            
            self.is_initialized = True
            self.logger.info("RAG agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG agent: {str(e)}")
            raise
    
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
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        embedding = self.embedding_model.embed_query(text)
        self.embedding_cache[text_hash] = embedding
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
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a document and create vector database
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document information dictionary
        """
        try:
            self.logger.info(f"Loading document: {file_path}")
            
            # Generate collection name
            collection_name = self.create_collection_name(file_path)
            
            try:
                # Load document and split into chunks
                self.logger.info("Processing document...")
                doc_splits = self.load_doc(file_path)
                
                # Check if document has any content
                if not doc_splits or len(doc_splits) == 0:
                    self.logger.error(f"Document has no extractable text: {file_path}")
                    return {
                        "success": False,
                        "error": "Document has no extractable text or is empty. Please upload a document with text content."
                    }
                
                try:
                    # Create FAISS vector database
                    self.logger.info(f"Creating FAISS vector database with {len(doc_splits)} chunks...")
                    vector_db = self.create_optimized_db(doc_splits, collection_name)
                    
                    # Create TF-IDF vectorizer for hybrid search
                    if self.use_hybrid_search:
                        self.logger.info("Creating TF-IDF vectorizer for hybrid search...")
                        tfidf_vectorizer = TfidfVectorizer()
                        tfidf_vectorizer.fit([doc.page_content for doc in doc_splits])
                        self.tfidf_vectorizers[collection_name] = tfidf_vectorizer
                    
                    # Save document info
                    document_info = {
                        "id": collection_name,
                        "path": file_path,
                        "name": os.path.basename(file_path),
                        "chunks": doc_splits,
                        "vector_db": vector_db
                    }
                    
                    # Store in documents dictionary
                    self.documents[collection_name] = document_info
                    
                    # Return the document info
                    return {
                        "document_id": collection_name,
                        "file_name": os.path.basename(file_path),
                        "chunks": len(doc_splits),
                        "success": True,
                        "message": f"Document uploaded and processed successfully with {len(doc_splits)} chunks."
                    }
                except Exception as vector_error:
                    self.logger.error(f"Error creating vector database: {str(vector_error)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return {
                        "success": False,
                        "error": f"Error creating vector database: {str(vector_error)}"
                    }
            except Exception as doc_error:
                self.logger.error(f"Error loading document: {str(doc_error)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {
                    "success": False,
                    "error": f"Error loading document: {str(doc_error)}"
                }
                
        except Exception as e:
            self.logger.error(f"Error in document processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_page(self, page: Document) -> List[Document]:
        """Process a single page into chunks
        
        Args:
            page: Document page
            
        Returns:
            List of document chunks
        """
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
    
    def load_doc(self, file_path: str) -> List[Document]:
        """Load PDF document with optimized chunking and parallel processing
        
        Args:
            file_path: Path to the PDF document
            
        Returns:
            List of document chunks
        """
        self.logger.info(f"Loading document: {file_path}")
        
        # Load document
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        if not self.use_parallel or len(pages) < 5:  # Only use parallel for larger documents
            self.logger.info(f"Standard processing for document with {len(pages)} pages")
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
        else:
            # Process pages in parallel
            self.logger.info(f"Parallel processing for document with {len(pages)} pages")
            chunks = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Process pages in parallel
                futures = [executor.submit(self.process_page, page) for page in pages]
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        page_chunks = future.result()
                        chunks.extend(page_chunks)
                    except Exception as e:
                        self.logger.error(f"Error processing page: {str(e)}")
            
            doc_splits = chunks
        
        self.logger.info(f"Document split into {len(doc_splits)} chunks")
        return doc_splits
    
    def create_optimized_db(self, splits: List[Document], collection_name: str) -> FAISS:
        """Create optimized FAISS index for better performance
        
        Args:
            splits: List of document chunks
            collection_name: Name for the FAISS index
            
        Returns:
            FAISS vector store
        """
        self.logger.info("Creating optimized FAISS vector database...")
        
        # Initialize the embedding model if not already initialized
        if not hasattr(self, 'embedding_model') or self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            )
        
        # Create FAISS index from documents - use standard method for small datasets
        if len(splits) < 100:
            self.logger.info(f"Using standard FAISS index for {len(splits)} chunks")
            vectordb = FAISS.from_documents(
                documents=splits,
                embedding=self.embedding_model
            )
        else:
            # Get document embeddings - use batched processing for larger datasets
            self.logger.info(f"Using optimized IVF FAISS index for {len(splits)} chunks")
            
            # Get embeddings
            texts = [doc.page_content for doc in splits]
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Convert to numpy array
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Create optimized FAISS index with IVF for faster search
            dimension = embeddings_np.shape[1]
            nlist = min(int(np.sqrt(len(splits))), 100)  # Rule of thumb: nlist ~= sqrt(n)
            
            # Create and train the index
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Need to train index
            if len(splits) >= nlist:
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
                docstore=FAISS.InMemoryDocstore(dict(zip(range(len(splits)), splits))),
                index_to_docstore_id=dict(zip(range(len(splits)), range(len(splits))))
            )
        
        # Save the FAISS index to disk
        index_path = os.path.join(self.faiss_index_dir, collection_name)
        vectordb.save_local(index_path)
        self.logger.info(f"FAISS vector database created and saved to {index_path}")
        
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
            
        # Get document chunks and vector db
        chunks = self.documents[document_id]["chunks"]
        vector_db = self.documents[document_id]["vector_db"]
        
        # Get TF-IDF vectorizer or create one if it doesn't exist
        if document_id not in self.tfidf_vectorizers:
            self.logger.info(f"Creating TF-IDF vectorizer for document {document_id}")
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_vectorizer.fit([chunk.page_content for chunk in chunks])
            self.tfidf_vectorizers[document_id] = tfidf_vectorizer
        else:
            tfidf_vectorizer = self.tfidf_vectorizers[document_id]
        
        # Keyword search with TF-IDF
        tfidf_matrix = tfidf_vectorizer.transform([chunk.page_content for chunk in chunks])
        query_vec = tfidf_vectorizer.transform([query])
        keyword_scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
        
        # Semantic search with embeddings
        semantic_results = vector_db.similarity_search_with_score(query, k=k)
        
        # Map semantic results to indices
        semantic_indices = []
        semantic_scores = []
        for doc, score in semantic_results:
            # Find index of the document in chunks
            for i, chunk in enumerate(chunks):
                if chunk.page_content == doc.page_content:
                    semantic_indices.append(i)
                    # Convert distance to similarity (1.0 - normalized distance)
                    semantic_scores.append(1.0 - min(score / 10.0, 1.0))
                    break
        
        # Combine scores (weighted sum of semantic and keyword)
        combined_scores = np.zeros(len(chunks))
        
        # Set semantic scores
        for idx, score in zip(semantic_indices, semantic_scores):
            combined_scores[idx] += self.semantic_weight * score
        
        # Add keyword scores (normalized)
        if np.max(keyword_scores) > 0:
            normalized_keyword_scores = keyword_scores / np.max(keyword_scores)
            combined_scores += self.keyword_weight * normalized_keyword_scores
        
        # Get top k results
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        
        return [chunks[i] for i in top_indices]
    
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
    
    def initialize_llmchain(self, document_id: str) -> bool:
        """Initialize the LLM chain for a specific document
        
        Args:
            document_id: Document ID to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if document exists
            if document_id not in self.documents:
                self.logger.error(f"Document ID not found: {document_id}")
                return False
            
            # Get vector database for document
            vector_db = self.documents[document_id]["vector_db"]
            
            self.logger.info(f"Initializing LLM chain with model: {self.model_id}")
            
            # Initialize the LLM
            llm = HuggingFaceEndpoint(
                repo_id=self.model_id,
                task="text-generation",
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                top_k=self.top_k,
                huggingfacehub_api_token=self.api_key,
            )

            self.logger.info("Setting up conversation memory...")
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                output_key="answer", 
                return_messages=True
            )
            
            # Configure the retriever
            retriever = self.setup_retriever(vector_db)

            self.logger.info("Creating RAG chain...")
            rag_prompt = PromptTemplate(
                template=self.SYSTEM_PROMPT, 
                input_variables=["context", "question"]
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=retriever,
                chain_type="stuff",
                memory=memory,
                combine_docs_chain_kwargs={"prompt": rag_prompt},
                return_source_documents=True,
                verbose=False,
            )
            
            self.logger.info("LLM chain initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM chain: {str(e)}")
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
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input
        
        Args:
            input_text: Input text for the agent
            callback: Optional callback for streaming responses
            
        Returns:
            Agent output text
        """
        try:
            # Check if this is a command in JSON format
            try:
                command = json.loads(input_text)
                action = command.get("action", "")
                
                if action == "upload":
                    # Upload a document
                    file_path = command.get("file_path", "")
                    if not file_path:
                        return json.dumps({
                            "success": False,
                            "error": "No file path provided"
                        })
                    
                    if callback:
                        callback("Uploading and processing document...")
                    
                    # Load the document
                    result = self.load_document(file_path)
                    
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
                    
                    # Initialize QA chain if needed
                    if self.qa_chain is None or document_id != self.current_document_id:
                        self.current_document_id = document_id
                        success = self.initialize_llmchain(document_id)
                        if not success:
                            return json.dumps({
                                "success": False,
                                "error": f"Failed to initialize LLM chain for document {document_id}"
                            })
                    
                    # Format history for the chain
                    chat_history = []
                    for entry in self.history:
                        chat_history.append((entry["user_input"], entry["agent_output"]))
                    
                    formatted_chat_history = self.format_chat_history(chat_history)
                    
                    # Use hybrid search if enabled
                    if self.use_hybrid_search and query:
                        self.logger.info(f"Using hybrid search for query: {query}")
                        
                        # Perform hybrid search
                        try:
                            retrieved_docs = self.hybrid_search(query, document_id, k=self.top_k)
                            
                            # Create a custom response with the retrieved documents
                            response = {
                                "answer": "I'm searching through the document to find relevant information...",
                                "source_documents": retrieved_docs
                            }
                            
                            # Generate response with LLM using retrieved documents as context
                            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            llm_chain = self.qa_chain.combine_docs_chain.llm_chain 
                            
                            prompt = self.SYSTEM_PROMPT.format(
                                question=query,
                                context=context
                            )
                            
                            answer = llm_chain.run(context=context, question=query)
                            response["answer"] = answer
                        except Exception as e:
                            self.logger.error(f"Error in hybrid search: {str(e)}")
                            # Fall back to standard retrieval
                            response = self.qa_chain.invoke({
                                "question": query, 
                                "chat_history": formatted_chat_history
                            })
                    else:
                        # Invoke QA chain with standard retrieval
                        response = self.qa_chain.invoke({
                            "question": query, 
                            "chat_history": formatted_chat_history
                        })
                    
                    # Extract answer and sources
                    response_answer = response["answer"]
                    response_sources = response["source_documents"]
                    
                    if response_answer.find("Helpful Answer:") != -1:
                        response_answer = response_answer.split("Helpful Answer:")[-1]
                    
                    # Format sources
                    sources = []
                    for i, doc in enumerate(response_sources[:self.top_k]):
                        page_num = doc.metadata.get("page", 0) + 1
                        content = doc.page_content.strip()
                        sources.append({
                            "page": page_num,
                            "content": content[:300] + ("..." if len(content) > 300 else "")
                        })
                    
                    # Add to history
                    self.add_to_history(query, response_answer)
                    
                    # Return JSON result
                    return json.dumps({
                        "success": True,
                        "answer": response_answer,
                        "sources": sources
                    })
                
                # Add custom action for updating agent configuration
                elif action == "update_config":
                    config_updates = command.get("config", {})
                    
                    # Update configuration parameters
                    for key, value in config_updates.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                            self.logger.info(f"Updated config parameter: {key} = {value}")
                    
                    # Reset the QA chain to apply new configuration
                    self.qa_chain = None
                    
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
            return f"Error: {str(e)}"
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
        self.qa_chain = None
        self.current_document_id = None
        
        # Clear caches to free memory
        self.embedding_cache.clear()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        capabilities = [
            "document_retrieval",
            "question_answering", 
            "pdf_processing"
        ]
        
        if self.use_hybrid_search:
            capabilities.append("hybrid_search")
        
        if self.use_dynamic_chunking:
            capabilities.append("dynamic_chunking")
            
        if self.use_mmr:
            capabilities.append("diverse_retrieval")
            
        return capabilities