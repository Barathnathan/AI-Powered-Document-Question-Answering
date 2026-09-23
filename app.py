import streamlit as st
from rag_pipeline import create_qa_chain

st.set_page_config(page_title="📚 Gemini RAG Q&A", layout="centered")
st.title("🔍 Ask Questions from Your PDFs (RAG + LLM)")

st.markdown("""
This app allows you to ask questions based on content inside your uploaded PDFs.  
It uses **FAISS**, **HuggingFace Embeddings**, and **Gemini 2.5 Flash** for high-speed, intelligent responses.
""")

# Load QA system
with st.spinner("🔄 Loading your AI assistant..."):
    qa_chain = create_qa_chain()
st.success("✅ Ready to answer questions!")

# Get user question
query = st.text_input("🧠 Ask your question here:")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            response = qa_chain.run(query)
            st.markdown("### 📘 Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
