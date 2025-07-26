import google.generativeai as genai
from config import GENERATION_MODEL

class AnswerGenerator:
    def __init__(self):
        self.model_name = GENERATION_MODEL
    
    def generate_answer(self, query, retrieved_chunks):
        """Generate answer based on retrieved chunks"""
        context = "\n".join([chunk[0] for chunk in retrieved_chunks])
        
        prompt = f"""You are a multilingual assistant. Answer in the same language as the query.
Use the following extracted texts:
{context}

Query: {query}

Answer:
"""
        
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        
        return response.text