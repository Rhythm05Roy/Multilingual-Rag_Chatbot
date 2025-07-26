from sklearn.metrics.pairwise import cosine_similarity
from embeddings import EmbeddingGenerator
from config import EMBEDDING_MODEL, DEFAULT_TOP_K

class DocumentRetriever:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
    
    def retrieve_chunks(self, query, vector_database, n_top=DEFAULT_TOP_K):
        """Retrieve most relevant chunks for a query"""
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        similarity_scores = []
        for entry in vector_database:
            chunk = entry["text"]
            chunk_embedding = entry["embedding"]
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            similarity_scores.append((chunk, similarity))
        
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        return similarity_scores[:n_top]