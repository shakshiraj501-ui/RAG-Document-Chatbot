# 🤖 RAG Document Chatbot

A simple Retrieval-Augmented Generation (RAG) based document chatbot built using Python, Streamlit, Sentence Transformers and FAISS.

## 📌 Features

- Upload PDF documents
- Extract text from PDF
- Divide PDF into smaller chunks
- Generate text embeddings
- Store embeddings using FAISS
- Search relevant information from the PDF
- Ask questions about the uploaded document

## 🛠️ Technologies Used

- Python
- Streamlit
- PyPDF
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- NumPy

## ⚙️ How It Works

1. **Upload PDF**  
   The user uploads a PDF document.

2. **Text Extraction**  
   The system extracts text from all pages of the PDF using PyPDF.

3. **Text Chunking**  
   The extracted text is divided into smaller chunks using LangChain Text Splitter.

4. **Generate Embeddings**  
   Sentence Transformers converts each text chunk into numerical vectors called embeddings.

5. **Create Vector Database**  
   FAISS stores these embeddings and allows fast similarity search.

6. **Ask a Question**  
   The user enters a question related to the uploaded PDF.

7. **Similarity Search**  
   The question is converted into an embedding and FAISS finds the most relevant chunks.

8. **Display Relevant Information**  
   The application displays the relevant information retrieved from the PDF.

### 🔄 RAG Workflow

PDF → Text Extraction → Chunking → Embeddings → FAISS → Question → Similarity Search → Relevant Information

## 🚀 How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
