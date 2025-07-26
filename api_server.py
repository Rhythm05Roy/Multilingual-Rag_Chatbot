from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import uvicorn

app = FastAPI(title="Multilingual RAG API", version="1.0.0")

# Global RAG pipeline instance
rag_pipeline = None

class QueryRequest(BaseModel):
    query: str

class PDFProcessRequest(BaseModel):
    pdf_path: str

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    try:
        rag_pipeline = RAGPipeline()
        print("✅ RAG Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG Pipeline: {e}")

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Multilingual RAG API is running!"}

@app.post("/process_pdf")
async def process_pdf(request: PDFProcessRequest):
    """Process PDF and create vector database"""
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        chunks_count = rag_pipeline.load_and_process_pdf(request.pdf_path)
        return {
            "message": "PDF processed successfully",
            "chunks_created": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process query and return answer"""
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        answer = rag_pipeline.query(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/database_info")
async def get_database_info():
    """Get information about the loaded database"""
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    return rag_pipeline.get_database_info()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)