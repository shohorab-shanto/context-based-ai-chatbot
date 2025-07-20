import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 3. Load and chunk PDF documents ---
pdf1 = PyPDFLoader("pdfs/topic1.pdf")   # Adjust your PDF paths accordingly
pdf2 = PyPDFLoader("pdfs/topic2.pdf")
docs1 = pdf1.load_and_split()
docs2 = pdf2.load_and_split()
docs = docs1 + docs2

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --- 4. Embed and store in FAISS ---
embedder = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = FAISS.from_documents(chunks, embedder)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- 5. LLM Setup (OpenAI-compatible proxy endpoint) ---
os.environ['GITHUB_TOKEN'] = "your_git_token"  # replace with your own token securely!
token = os.environ.get("GITHUB_TOKEN")
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-nano"

llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=token,
    openai_api_base=endpoint,
    temperature=0.3,
)

# --- 6. Prompt setup for memory-aware Q&A ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Turn the user query into a standalone question using the chat history."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use only the context below to answer the user's question. Be concise.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# --- 7. Build RAG chain with memory ---
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
document_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

# --- 8. Chat history manager class ---
class SessionHistoryManager:
    def __init__(self):
        self.store = {}

    def get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    # def reset(self, session_id: str):
    #     self.store.pop(session_id, None)

history_manager = SessionHistoryManager()

# --- 9. Final chain with memory tracking ---
chatbot = RunnableWithMessageHistory(
    rag_chain,
    history_manager.get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# --- FastAPI app setup ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": []})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    # Use a fixed session_id for demo, you can make this dynamic (cookie, user auth, etc.)
    session_id = "user_session"

    # Run the chatbot with session memory
    response = chatbot.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

# @app.post("/reset", response_class=RedirectResponse)
# async def reset_session(request: Request):
#     session_id = "user_session"  # Same as in /chat
#     history_manager.store.pop(session_id, None)  # Delete the history
#     return RedirectResponse(url="/", status_code=303)

    # Fetch current chat history to display
    history = history_manager.get_history(session_id)

    # Prepare list of messages for UI rendering
    chat_messages = []
    for msg in history.messages:
        # msg.type will be 'human' for user, 'ai' for bot
        role = "user" if msg.type == "human" else "bot"
        chat_messages.append({"role": role, "content": msg.content})

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": chat_messages,
    })
