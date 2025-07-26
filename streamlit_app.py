import streamlit as st
import os
from rag_pipeline import RAGPipeline
import tempfile
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Multilingual RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

def initialize_rag_pipeline():
    """Initialize RAG pipeline with error handling"""
    try:
        st.session_state.rag_pipeline = RAGPipeline()
        return True
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        return False

def process_pdf(uploaded_file):
    """Process uploaded PDF file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process PDF
        with st.spinner("Processing PDF... This may take a few minutes."):
            chunks_count = st.session_state.rag_pipeline.load_and_process_pdf(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        st.session_state.pdf_processed = True
        return chunks_count
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def main():
    st.title("🤖 Multilingual RAG Chatbot")
    st.markdown("### Especially for Bengali and English Languages")
    
    # Sidebar for settings only
    with st.sidebar:
        st.header("⚙️ Settings")
        # Initialize RAG pipeline if not already done
        if st.session_state.rag_pipeline is None:
            if st.button("Initialize System"):
                if initialize_rag_pipeline():
                    st.success("System initialized successfully!")
                    st.rerun()
        # Reset database button (if processed)
        if st.session_state.pdf_processed:
            if st.button("Reset Database"):
                st.session_state.pdf_processed = False
                st.session_state.chat_history = []
                st.session_state.rag_pipeline = None
                st.success("Database reset successfully!")
                st.rerun()

    # Main chat interface
    if st.session_state.rag_pipeline is None:
        st.info("👈 Please initialize the system using the sidebar.")
        return

    st.header("💬 Chat Interface")

    # PDF upload and process in chat section
    if not st.session_state.pdf_processed:
        st.subheader("📁 Upload and Process PDF Document")
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=['pdf'],
            help="Upload a PDF document to create a knowledge base"
        )
        if uploaded_file is not None:
            if st.button("Process PDF"):
                chunks_count = process_pdf(uploaded_file)
                if chunks_count:
                    st.success(f"PDF processed successfully! Created {chunks_count} chunks.")
                    st.rerun()
        st.info("Please upload and process a PDF document to start chatting.")
        return

    # Database info (after processing)
    st.subheader("📊 Database Info")
    db_info = st.session_state.rag_pipeline.get_database_info()
    st.info(f"Total chunks: {db_info['total_chunks']}")
    st.info(f"Total characters: {db_info['total_characters']:,}")

    # Sample queries
    st.subheader("💡 Sample Queries")
    sample_queries = [
        "অনুপমের বন্ধুর নাম কি?",
        "অপরিচিত গল্পের লেখক কে?",
        "কল্যাণীর পিতার নাম কি?",
        "What is the main theme?",
        "মামা কোন ঘরের মেয়ে পছন্দ করতেন?",
        "Who is the main character?",
    ]
    for query in sample_queries:
        if st.button(query, key=f"sample_{hash(query)}"):
            if st.session_state.pdf_processed:
                # Add to chat history and trigger response immediately
                st.session_state.chat_history.append({"role": "user", "content": query})
                # Generate response
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_pipeline.query(query)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question in Bengali or English..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_pipeline.query(prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()