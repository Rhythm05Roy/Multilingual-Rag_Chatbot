#!/bin/bash

# Setup script for the Multilingual RAG Chatbot project

echo "🚀 Setting up Multilingual RAG Chatbot..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update

# Install Tesseract OCR and Bengali language support
echo "🔤 Installing Tesseract OCR with Bengali support..."
sudo apt install tesseract-ocr tesseract-ocr-ben -y

# Install Poppler utilities for PDF processing
echo "📄 Installing Poppler utilities..."
sudo apt-get install poppler-utils -y

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Update API keys in config.py"
echo "2. Run the Streamlit app: streamlit run streamlit_app.py"
echo "3. Or run the FastAPI server: python api_server.py"
echo ""
echo "📚 Usage:"
echo "- Streamlit UI: Interactive web interface for chatting"
echo "- FastAPI: REST API for programmatic access"
echo "- Evaluation: Use evaluation.py to test the system"