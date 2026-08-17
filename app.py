import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("🤖 RAG Document Chatbot")

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

    # Load embedding model
    with st.spinner("Creating embeddings..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True
        )

    # Create FAISS index
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

        question_embedding = model.encode(
            [question],
            convert_to_numpy=True
        ).astype("float32")

        distances, indices = index.search(
            question_embedding,
            k=3
        )

        st.subheader("🔍 Relevant Information")

        for i in indices[0]:
            st.write(chunks[i])
            st.write("---")