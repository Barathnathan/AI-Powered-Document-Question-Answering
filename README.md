# 📚 AI-Powered PDF Question Answering System

An AI-powered **PDF Question Answering System** built using **Retrieval-Augmented Generation (RAG)**. The application allows users to ask questions about information contained in PDF documents and generates answers using relevant content retrieved from those documents.

The project combines **LangChain, FAISS, HuggingFace Embeddings, Google Gemini, PyPDFLoader, and Streamlit** to create an end-to-end document question-answering application.

---

## 🚀 Project Overview

Traditional Large Language Models may not have access to information contained in a user's private documents.

This project uses **Retrieval-Augmented Generation (RAG)** to connect a language model with information extracted from PDF documents.

The system:

1. Loads PDF documents from the `data/` directory.
2. Extracts text from the PDFs.
3. Splits the extracted text into smaller chunks.
4. Converts the chunks into vector embeddings using HuggingFace.
5. Stores the embeddings in a FAISS vector store.
6. Retrieves relevant document content when a user asks a question.
7. Sends the retrieved information to Google Gemini.
8. Generates an answer through an interactive Streamlit interface.

---

## ✨ Features

- 📄 PDF document processing
- 🔎 Semantic document retrieval
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤗 HuggingFace sentence embeddings
- ⚡ FAISS vector similarity search
- 🤖 Google Gemini integration
- 🔗 LangChain-based RAG pipeline
- 🖥️ Streamlit web interface
- 💾 Persistent FAISS vector store
- 🔐 Environment variable support for API keys
- 🧩 Modular Python implementation

---

## 🧠 RAG Architecture

```text
                         ┌──────────────────┐
                         │   PDF Documents  │
                         │    data/*.pdf    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  PyPDFLoader     │
                         │  Load PDF Text   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Text Splitter    │
                         │ Chunk Size: 500  │
                         │ Overlap: 50      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ HuggingFace      │
                         │ Embeddings       │
                         │ all-MiniLM-L6-v2 │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ FAISS Vector     │
                         │ Store            │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  User Question   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Document         │
                         │ Retrieval        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Google Gemini    │
                         │ Gemini 2.5 Pro   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Generated Answer │
                         └──────────────────┘
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web application interface |
| **LangChain** | RAG pipeline and LLM orchestration |
| **Google Gemini** | Large Language Model |
| **HuggingFace** | Text embedding generation |
| **FAISS** | Vector similarity search |
| **PyPDFLoader** | PDF document processing |
| **python-dotenv** | Environment variable management |
| **Pickle** | Local vector store persistence |

---

## 📁 Project Structure

```text
AI-Powered-PDF-Question-Answering/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── test.py
├── example.pdf
│
├── data/
│   └── your-document.pdf
│
└── vector_store/
    └── faiss_index.pkl
```

### Important

The following files/folders should **not** be uploaded to GitHub:

```text
.env
venv/
__pycache__/
.streamlit/
```

Make sure your `.gitignore` contains:

```text
.env
venv/
__pycache__/
.streamlit/
*.pyc
```

---

# 📄 File Descriptions

### `app.py`

The Streamlit application responsible for the user interface.

It:

- Configures the Streamlit page
- Displays the application title and description
- Loads the RAG question-answering chain
- Accepts questions from the user
- Sends questions to the QA chain
- Displays generated answers
- Handles errors during question answering

### `rag_pipeline.py`

Contains the core RAG implementation.

It handles:

- Loading environment variables
- Loading PDF documents
- Splitting documents into chunks
- Creating HuggingFace embeddings
- Creating/loading the FAISS vector store
- Creating the document retriever
- Connecting the retriever with Google Gemini
- Creating the LangChain `RetrievalQA` chain

### `data/`

Contains the PDF documents that the application processes.

Example:

```text
data/
└── example.pdf
```

### `vector_store/`

Contains the locally persisted FAISS vector store.

---

# 🔄 How the RAG Pipeline Works

## 1. Load PDF Documents

PDF files are loaded using LangChain's `PyPDFLoader`.

```python
loader = PyPDFLoader(os.path.join(folder_path, file_name))
```

The extracted pages are collected into a document list.

---

## 2. Split Documents

The extracted text is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### Current Configuration

```text
Chunk Size    → 500
Chunk Overlap → 50
```

This helps the retrieval system find relevant sections of the document.

---

## 3. Generate Embeddings

The project uses the HuggingFace embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts document chunks into numerical vector representations.

---

## 4. Create FAISS Vector Store

FAISS is used to store and search the generated embeddings.

If a vector store already exists, the application loads the existing vector store.

Otherwise, a new FAISS index is created from the document chunks.

---

## 5. Retrieve Relevant Information

When a user enters a question, the FAISS retriever searches for document chunks that are semantically related to the question.

The retrieved information is provided to the language model as context.

---

## 6. Generate the Answer

The retrieved documents are passed to Google Gemini through LangChain's `RetrievalQA` chain.

The current model configuration is:

```python
ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2
)
```

Gemini then generates the final response using the retrieved document context.

---

# 🖥️ Application Interface

The application provides a simple Streamlit interface.

Users can enter their question using:

```text
🧠 Ask your question here:
```

The system processes the question and displays:

```text
📘 Answer:
```

The application also displays loading indicators while the RAG pipeline and answer generation are running.

---

# ⚙️ Installation

## Prerequisites

Make sure you have:

- Python 3.10+
- Git
- Google Gemini API key

---

## 1. Clone the Repository

```bash
git clone https://github.com/Barathnathan/AI-Powered-PDF-Question-Answering.git
```

Move into the project directory:

```bash
cd AI-Powered-PDF-Question-Answering
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure Google Gemini API

The application uses Google Gemini for answer generation.

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Replace `your_google_gemini_api_key` with your actual Gemini API key.

### 🔐 Security

Never upload your API key to GitHub.

Your `.gitignore` should contain:

```text
.env
venv/
__pycache__/
.streamlit/
*.pyc
```

---

# 📚 Add PDF Documents

Place your PDF documents inside the `data` directory.

For example:

```text
data/
└── example.pdf
```

The application automatically searches the `data/` directory for PDF files.

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Alternatively:

```bash
streamlit run app.py
```

After starting the application, Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 💬 Example Usage

Suppose the PDF contains information about Machine Learning.

You could ask:

```text
What is supervised learning?
```

The application follows this process:

```text
User Question
      ↓
FAISS searches document embeddings
      ↓
Relevant PDF content retrieved
      ↓
Retrieved content passed to Gemini
      ↓
Gemini generates the answer
      ↓
Answer displayed in Streamlit
```

---

# 🧩 Core Components

### Document Loader

**PyPDFLoader**

Used to load and extract content from PDF documents.

### Text Splitter

**RecursiveCharacterTextSplitter**

Used to divide documents into smaller chunks.

### Embedding Model

**sentence-transformers/all-MiniLM-L6-v2**

Used to convert document text into vector representations.

### Vector Store

**FAISS**

Used for similarity-based document retrieval.

### Language Model

**Google Gemini 2.5 Pro**

Used to generate answers based on retrieved document context.

### Application Framework

**Streamlit**

Used to build the interactive web application.

---

# 🎯 Learning Objectives

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Vector embeddings
- Semantic search
- Vector databases
- PDF document processing
- LangChain
- HuggingFace embeddings
- Google Gemini API integration
- FAISS
- Streamlit
- Python environment management
- Environment variable management
- End-to-end AI application development

---

# 📈 Project Workflow

```text
                    PDF DOCUMENT
                         │
                         ▼
                 ┌───────────────┐
                 │ PDF Loading   │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Text Splitting│
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │   Embeddings  │
                 │  HuggingFace  │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ FAISS Vector  │
                 │     Store     │
                 └───────┬───────┘
                         │
                         ▼
                  USER QUESTION
                         │
                         ▼
                 ┌───────────────┐
                 │   Retriever   │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Relevant PDF  │
                 │    Context    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Google Gemini │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │    Answer     │
                 └───────────────┘
```

---

# 🔍 Why RAG?

Retrieval-Augmented Generation combines two major processes:

### Retrieval

Relevant information is retrieved from an external knowledge source such as PDF documents.

### Generation

A language model generates a natural-language response using the retrieved information.

This architecture is useful for document-based question answering because the application can provide document content to the language model as context during answer generation.

---

# ⚠️ Current Limitations

The current implementation has some limitations:

- PDFs need to be placed manually inside the `data/` directory.
- The application does not currently provide a PDF upload interface.
- Conversation history is not implemented.
- Source citations and page references are not currently displayed in the UI.
- The FAISS vector store is persisted locally.
- A Google Gemini API key is required for answer generation.

---

# 🔮 Future Improvements

Possible improvements for future versions include:

- 📤 Upload PDFs directly through the Streamlit interface
- 📚 Support multiple document collections
- 💬 Add conversation history
- 📍 Display PDF source and page references
- 🎛️ Add configurable retrieval parameters
- 🔎 Improve document search and retrieval
- 🔐 Add authentication and user management
- ☁️ Deploy the application to a cloud platform
- 📊 Add retrieval and answer evaluation metrics
- 🧠 Add support for additional LLM providers
- ⚡ Improve vector-store management
- 🗂️ Allow users to manage uploaded documents

---

# 🛡️ Security

API keys and other sensitive credentials should never be committed to the repository.

Use environment variables:

```env
GOOGLE_API_KEY=your_api_key
```

Keep `.env` in `.gitignore`.

---

# 🤝 Contributing

Contributions and improvements are welcome.

### Clone the repository

```bash
git clone https://github.com/Barathnathan/AI-Powered-PDF-Question-Answering.git
```

### Create a branch

```bash
git checkout -b feature/new-feature
```

### Make your changes

### Commit your changes

```bash
git add .
git commit -m "Add new feature"
```

### Push your branch

```bash
git push origin feature/new-feature
```

Then create a Pull Request.

---

# 📄 License

This project is available for educational and development purposes.

---

# 👨‍💻 Author

## Barath Nathan

GitHub:

https://github.com/Barathnathan

Project Repository:

https://github.com/Barathnathan/AI-Powered-PDF-Question-Answering

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 🚀 Built With

**Python • Streamlit • LangChain • Google Gemini • HuggingFace • FAISS • PyPDFLoader**

**Built as a practical implementation of Retrieval-Augmented Generation for document-based question answering.**
