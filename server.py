from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import time
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
    api_key: Optional[str] = None        # Optional client-supplied key
    provider: str = "gemini"             # "gemini" or "openai"
    active_category: Optional[str] = None # Optional client-supplied category context ("1"-"6")

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

# Endpoint to submit feedback
class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int  # 1 for positive, -1 for negative
    comment: Optional[str] = None
    category: Optional[str] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    feedback_file = "feedback.json"
    feedback_data = []
    
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r") as f:
                content = f.read().strip()
                if content:
                    feedback_data = json.loads(content)
        except Exception:
            feedback_data = []
            
    feedback_entry = {
        "timestamp": time.time(),
        "user_email": req.user_email,
        "user_role": req.user_role,
        "query": req.query,
        "response": req.response,
        "rating": req.rating,
        "comment": req.comment,
        "category": req.category
    }
    feedback_data.append(feedback_entry)
    
    try:
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
        
    return {"status": "success", "message": "Feedback submitted successfully"}

# Static Routing for the Web UI
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/style.css")
async def read_style():
    return FileResponse("static/style.css")

@app.get("/app.js")
async def read_app():
    return FileResponse("static/app.js")
