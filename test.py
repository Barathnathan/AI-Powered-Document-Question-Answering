import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Set embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Step 1: Load PDF documents
def load_documents(folder_path="data"):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

# Step 2: Split documents
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Step 3: Create or load FAISS vector store
def create_vector_store(splits, index_path="vector_store/faiss_index.pkl"):
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store = FAISS.from_documents(splits, embeddings)
        os.makedirs("vector_store", exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(vector_store, f)
    return vector_store

# Step 4: Create QA chain
def create_qa_chain():
    print("📄 Loading documents...")
    documents = load_documents()
    print(f"✅ Loaded {len(documents)} documents.")

    print("🔍 Splitting into chunks...")
    splits = split_documents(documents)
    print(f"✅ Created {len(splits)} chunks.")

    print("💾 Creating/loading FAISS index...")
    vector_store = create_vector_store(splits)

    retriever = vector_store.as_retriever()

    print("🤖 Initializing Gemini Pro...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Step 5: Local Testing (CLI Mode)
if __name__ == "__main__":
    qa_chain = create_qa_chain()
    print("\n🚀 RAG System Ready! Ask your questions below. Type 'exit' to quit.\n")

    while True:
        query = input("❓ Your Question: ")
        if query.lower() == "exit":
            break
        try:
            answer = qa_chain.run(query)
            print("📘 Answer:", answer, "\n")
        except Exception as e:
            print("❌ Error:", e)
