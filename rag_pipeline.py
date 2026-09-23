import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Create the embedding model explicitly
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load .env file
load_dotenv()

# Load PDF documents
def load_documents(folder_path="data"):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file_name))
            documents.extend(loader.load())
    return documents

# Split documents
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Create or load FAISS vector store
def create_vector_store(splits):
    embeddings = HuggingFaceEmbeddings()
    index_path = "vector_store/faiss_index.pkl"

    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store = FAISS.from_documents(splits, embeddings)
        os.makedirs("vector_store", exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(vector_store, f)

    return vector_store

# Create QA chain
def create_qa_chain():
    documents = load_documents()
    splits = split_documents(documents)
    vector_store = create_vector_store(splits)

    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
