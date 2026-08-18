import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai
import faiss
import numpy as np
import os

st.title("🤖 RAG Document Chatbot")

# Connect Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found.")
    st.stop()

client = genai.Client(api_key=api_key)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    st.success("PDF uploaded successfully! ✅")

    # Split PDF text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    st.success(f"PDF divided into {len(chunks)} chunks! ✅")

    # Create embeddings
    with st.spinner("Creating embeddings..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True
        )

    # Create FAISS vector database
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    st.success("Vector database created! ✅")

    # Ask question
    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if question:

        # Convert question to embedding
        question_embedding = model.encode(
            [question],
            convert_to_numpy=True
        ).astype("float32")

        # Search relevant chunks
        distances, indices = index.search(
            question_embedding,
            k=3
        )

        relevant_text = ""

        for i in indices[0]:
            relevant_text += chunks[i] + "\n\n"

        # Generate answer using Gemini
        with st.spinner("Generating answer... 🤖"):

            prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not present in the context, say:
"I could not find this information in the uploaded PDF."

Document Context:
{relevant_text}

User Question:
{question}

Give a clear and concise answer.
"""

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

        st.subheader("🤖 Answer")
        st.write(response.text)

        # Show retrieved information
        with st.expander("🔍 View Relevant Information"):
            st.write(relevant_text)