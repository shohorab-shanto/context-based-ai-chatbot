# 🧠 LangChain Chatbot with PDF Knowledge & Memory (FastAPI)

This is a memory-aware chatbot application powered by **LangChain**, **FAISS**, and **FastAPI**. It loads knowledge from two PDFs, builds a vector store, and allows users to interact with the content via a contextual chatbot interface. Chat history is preserved per session and can be reset easily.

---

## 🚀 Features

- Load multiple PDFs using `PyPDFLoader`
- Chunk text and embed using `HuggingFaceEmbeddings`
- Store vectors in `FAISS` for fast retrieval
- Ask follow-up or contextual questions with LangChain’s memory-enabled RAG pipeline
- Reset chat history via a simple UI button
- Lightweight frontend with HTML + Jinja2 templates

---

## 📁 Project Structure

📦 langchain-pdf-chatbot/
├── app.py # Main FastAPI app
├── templates/
│ └── index.html # Chat UI template
├── static/ # Optional static folder for CSS/JS
├── pdfs/
│ ├── topic1.pdf
│ └── topic2.pdf
├── requirements.txt
└── README.md


---

## 🛠️ Requirements

- Python 3.8+
- pip

---

## 📦 Installation


git clone https://github.com/your-username/langchain-pdf-chatbot.git
cd langchain-pdf-chatbot
python -m venv venv
source venv/bin/activate

## For Local setup command
pip install fastapi uvicorn jinja2 aiofiles      
pip install langchain langchain-openai langchain-community langchain-core faiss-cpu sentence-transformers unstructured pypdf

## PDF Setup
Place your two PDFs in the pdfs/ directory and name them:

topic1.pdf
topic2.pdf

These will be used as the knowledge base for the chatbot.

🔑 Configure LLM
This uses an OpenAI-compatible endpoint. You must set a token (e.g., from GitHub AI or similar):

os.environ['GITHUB_TOKEN'] = "your-token-here"

## Run the app:
uvicorn app:app --reload
