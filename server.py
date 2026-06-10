from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv

# Load keys from .env file
load_dotenv()

from agent import stream_agent

app = FastAPI(title="Stateless Multi-Provider FloCareer Chatbot Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    api_key: str = None        # Optional client-supplied key
    provider: str = "gemini"   # "gemini" or "openai"
    active_category: str = None # Optional client-supplied category context ("1"-"6")

# Unified LangGraph SSE Stream Generator
async def stateless_agent_generator(query: str, provider: str, api_key: str, active_cat: str):
    inputs = {
        "messages": [],
        "query": query,
        "provider": provider,
        "api_key": api_key,
        "active_category": active_cat
    }
    
    current_active_cat = active_cat
    async for chunk in stream_agent(inputs):
        if chunk["type"] in ["token", "text"]:
            yield f"data: {json.dumps({'text': chunk['text']})}\n\n"
        elif chunk["type"] == "active_category":
            current_active_cat = chunk["value"]
            
    # Yield the final active category back to the client
    yield f"data: {json.dumps({'text': '', 'active_category': current_active_cat})}\n\n"
    yield "data: [DONE]\n\n"

# Public Stateless Stream Endpoint
@app.post("/api/chat")
async def stateless_chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    provider = req.provider.lower()
    effective_api_key = None
    if provider == "openai":
        effective_api_key = req.api_key or os.environ.get("OPENAI_API_KEY")
    elif provider == "gemini":
        effective_api_key = req.api_key or os.environ.get("GEMINI_API_KEY")
        
    return StreamingResponse(
        stateless_agent_generator(
            req.message.strip(),
            provider if effective_api_key else "local",
            effective_api_key,
            req.active_category
        ),
        media_type="text/event-stream"
    )
