RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDF documents and ask questions based on their content.

Features
Upload PDF documents
Extract text from PDFs
Split text into chunks
Generate embeddings using Sentence Transformers
Store embeddings in FAISS Vector Database
Ask questions based on the uploaded document
User-friendly Streamlit interface
Gemini API integration for AI-generated responses
Technologies Used
Python
Streamlit
PyPDF
Sentence Transformers
FAISS
Google Gemini API
LangChain
How It Works
Upload a PDF document.
The application extracts text from the PDF.
The text is divided into smaller chunks.
Embeddings are generated using Sentence Transformers.
The embeddings are stored in a FAISS vector database.
When a user asks a question, the application finds relevant document chunks.
Gemini generates an answer using the retrieved context.
Installation

Clone the repository and install the required dependencies:

pip install -r requirements.txt

Run the application:

python -m streamlit run app.py
Project Structure
RAG_Application/
│── app.py
│── requirements.txt
│── README.md
│── venv/
Author

Sakshi Singh
