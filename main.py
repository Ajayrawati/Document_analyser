import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  
from docx import Document


st.title("Document Analyzer")
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload TXT, PDF, DOCX files", type=["txt", "pdf", "docx"], accept_multiple_files=True
)


embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = None
documents = []


def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


if st.sidebar.button("Process Documents"):
    corpus = []
    for file in uploaded_files:
        file_type = file.name.split('.')[-1].lower()

        if file_type == "txt":
            content = file.read().decode("utf-8")
        elif file_type == "pdf":
            content = extract_text_from_pdf(file)
        elif file_type == "docx":
            content = extract_text_from_docx(file)
        else:
            continue
        
        corpus.append(content)
        documents.append(content)

    if corpus:
        st.write("Building FAISS index...")
        embeddings = embedder.encode(corpus, convert_to_tensor=False)
        embeddings = np.array(embeddings)
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        st.success(f"Index created with {len(documents)} documents!")

user_query = st.text_input("Enter your question:")

if index is not None and len(documents) > 0:
    st.header("Ask Questions about Your Documents")

    

    if user_query:
        st.write(f"Processing your query: '{user_query}'")
      
        query_embedding = embedder.encode(user_query, convert_to_tensor=False)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        D, I = index.search(query_embedding, k=3)

        if I[0].size > 0:
            relevant_docs = [documents[i] for i in I[0]]
            context = "\n".join(relevant_docs[:3])
            st.write("Relevant context extracted.")

            try:
                qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
                
                result = qa_pipeline({
                    "context": context,
                    "question": user_query
                })
                
                st.write("*Extracted Answer:*", result["answer"])
            except Exception as e:
                st.error(f"Error during question answering: {e}")
        else:
            st.warning("No relevant documents found for the query.")
