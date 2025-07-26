from pdf_processor import PDFProcessor
from embeddings import EmbeddingGenerator
from retrieval import DocumentRetriever
from generation import AnswerGenerator
from config import configure_genai, MAX_CHUNK_WORDS

class RAGPipeline:
    def __init__(self, api_key_index=0):
        # Configure API
        configure_genai(api_key_index)
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.retriever = DocumentRetriever()
        self.answer_generator = AnswerGenerator()
        
        # Storage
        self.vector_database = None
        self.processed_text = None
    
    def load_and_process_pdf(self, pdf_path):
        """Load and process PDF to create vector database"""
        # Process PDF
        text, chunks = self.pdf_processor.process_pdf(pdf_path, MAX_CHUNK_WORDS)
        self.processed_text = text
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        
        # Create vector database
        self.vector_database = self.embedding_generator.create_vector_database(chunks, embeddings)
        
        return len(chunks)
    
    def query(self, question):
        """Query the RAG system"""
        if self.vector_database is None:
            return "Please load and process a PDF first."
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve_chunks(question, self.vector_database)
        
        # Generate answer
        answer = self.answer_generator.generate_answer(question, retrieved_chunks)
        
        return answer
    
    def get_database_info(self):
        """Get information about the loaded database"""
        if self.vector_database is None:
            return "No database loaded"
        
        return {
            "total_chunks": len(self.vector_database),
            "total_characters": len(self.processed_text) if self.processed_text else 0
        }