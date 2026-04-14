from typing import List, Dict, Any
import os

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding

class LlamaIndexPipeline:
    """
    A pipeline class that implements the 4 stages of RAG with LlamaIndex:
    1. Ingestion
    2. Chunking
    3. Embedding
    4. Retrieval
    
    The research agent simply calls process_and_retrieve with the articles and query.
    """
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        """
        Initializes the pipeline with configurable parameters for chunking and retrieval.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Configure global settings for chunking
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Configure Gemini Embedding model
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=api_key)
        else:
            print("WARNING: GEMINI_API_KEY not found in environment. Default embeddings will be used if any.")

    def process_and_retrieve(self, articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Takes external document dictionaries, builds an in-memory index, and queries it.
        
        :param articles: List of dictionaries representing PubMed articles.
                         e.g., [{"title": "...", "abstract": "...", "pmid": "123"}]
        :param query: The illness query string to search for (e.g., "Hashimoto's thyroiditis").
        :return: List of relevant chunks with their metadata.
        """
        if not articles:
            return []

        # Stage 1: Ingestion
        # Convert raw dictionaries to LlamaIndex Document objects
        documents = []
        for article in articles:
            # We assume title and content carry the main information (as output by pubmed_tool.py)
            title = article.get("title", "No Title")
            content = article.get("content", "")
            
            # Combine into a single text block
            text = f"Title: {title}\n\nContent:\n{content}"
            
            # Extract metadata (excluding the main text fields to avoid duplication)
            metadata = {k: v for k, v in article.items() if k not in ["content"]}
            
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)

        # Stage 2 & 3: Chunking & Embedding
        # VectorStoreIndex.from_documents handles applying the node parser (chunking)
        # and calling the embedding model to build the in-memory vector store.
        index = VectorStoreIndex.from_documents(documents)

        # Stage 4: Retrieval
        # Retrieve the top_k most similar chunks to the user's query
        retriever = index.as_retriever(similarity_top_k=self.top_k)
        retrieved_nodes = retriever.retrieve(query)

        # Format and return the relevant chunks for the research agent
        results = []
        for node in retrieved_nodes:
            results.append({
                "chunk_text": node.get_text(),
                "metadata": node.metadata,
                "relevance_score": node.score
            })

        return results
