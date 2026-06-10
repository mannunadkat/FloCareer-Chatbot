# FloCareer AI Assistant & RAG Chatbot

An intelligent conversational support agent for FloCareer, powered by Retrieval-Augmented Generation (RAG) and structured as a state machine using **LangGraph** and **LangChain**.

---

## 🚀 Key Features

- **LangGraph State Machine**: Conversation flows (greetings, categorizations, option selection, RAG search, LLM routing, and local fallback) are modeled as a robust StateGraph inside `agent.py`.
- **Hybrid Search RAG**: Combines local keyword tokenization and cosine-similarity semantic search using **ChromaDB** and **SentenceTransformers** (`all-MiniLM-L6-v2`).
- **Interactive Menu Flow**: Allows users to navigate topics sequentially via category numbers (`1`-`6`) and options (`a`-`h`).
- **Robust Auto-Spelling Correction**: Reconstructs misspelled user queries using Damerau-Levenshtein distance mapping against vocabulary compiled from the FAQ sheets.
- **Multi-Provider Support**: Supports streaming responses via **OpenAI (GPT-4o-mini)**, **Gemini (Gemini 2.5 Flash)**, and an offline local RAG fallback.
- **Stateless SSE Streaming Server**: Serves responses as a Server-Sent Events (SSE) stream via **FastAPI** to easily connect with web frontends.
- **Premium CLI Portal**: Features a styled interactive command-line interface with text animations and theme highlights.

---

## 📁 Project Structure

```text
├── chroma_db/             # Local Chroma DB persistent storage
├── agent.py               # LangGraph state graph definitions, nodes, and streaming api
├── chatbot.py             # Premium CLI portal entry point
├── server.py              # FastAPI server exposing the SSE streaming endpoint
├── rag.py                 # RAG engine (parser, vocab builder, typo corrector, ChromaDB setup)
├── knowledge_base.md      # Parsed FAQ entries markdown compiled from excel sheets
├── .env                   # Local API keys (OpenAI / Gemini)
└── venv/                  # Python virtual environment
```

---

## 🛠️ Getting Started

### 1. Setup Environment
Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/mannunadkat/FloCareer-Chatbot.git
cd FloCareer-Chatbot

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```
*(If no `requirements.txt` exists, install core dependencies: `pip install fastapi uvicorn chromadb sentence-transformers langchain-core langchain-community langchain-openai langchain-google-genai langgraph requests python-dotenv openpyxl`)*

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys:

```ini
OPENAI_API_KEY="your-openai-api-key"
# OR
GEMINI_API_KEY="your-gemini-api-key"
```
*If no keys are provided, the chatbot automatically falls back to local RAG search mode.*

---

## 💻 Running the Chatbot

### CLI Portal
To chat with the bot in your terminal, run:
```bash
python chatbot.py
```

### FastAPI SSE Server
To launch the backend API server:
```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

---

## 🔌 API Endpoint Reference

### Chat Stream Endpoint
`POST /api/chat`

Exposes a Server-Sent Events (SSE) streaming API.

#### Request Body
```json
{
  "message": "hi",
  "provider": "openai",
  "api_key": "optional-override-api-key",
  "active_category": null
}
```

#### Response Stream (Chunks)
```text
data: {"text": "Hey there! 👋 Welcome to FloCareer Support."}

data: {"text": "", "active_category": null}

data: [DONE]
```
