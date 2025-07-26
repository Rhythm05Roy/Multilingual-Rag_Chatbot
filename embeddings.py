import google.generativeai as genai
import numpy as np
from config import EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self):
        self.embedding_model = EMBEDDING_MODEL
    
    def generate_embeddings(self, text_chunks):
        """Generate embeddings for text chunks"""
        chunk_embeddings = []
        for chunk in text_chunks:
            embedding = genai.embed_content(
                model=self.embedding_model, 
                content=chunk, 
                task_type="retrieval_document"
            )
            chunk_embeddings.append(embedding['embedding'])
        
        print(f"✅ Generated {len(chunk_embeddings)} embeddings")
        return chunk_embeddings
    
    def create_vector_database(self, text_chunks, embeddings):
        """Create in-memory vector database"""
        vector_database = []
        for i, chunk in enumerate(text_chunks):
            vector_database.append({
                "text": chunk,
                "embedding": embeddings[i]
            })
        
        print(f"✅ Created in-memory database with {len(vector_database)} entries")
        return vector_database
    
    def generate_query_embedding(self, query):
        """Generate embedding for query"""
        query_embedding = genai.embed_content(
            model=self.embedding_model, 
            content=query, 
            task_type="retrieval_query"
        )['embedding']
        return query_embedding