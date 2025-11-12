import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import db, create_document, get_documents
from schemas import Material, Flashcard, QuizItem, Plan, Performance, ChatHistory

# Minimal, dependency-light retrieval
import numpy as np

# PDF parsing
import fitz  # PyMuPDF

# Optional LLMs with graceful fallback
try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from bson import ObjectId

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        genai = None

openai_client = None
if OpenAI and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None

app = FastAPI(title="VectorTutor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadResponse(BaseModel):
    material_id: str
    topics: List[str]

class AskRequest(BaseModel):
    user_id: str
    material_id: str
    question: str

class QuizSubmit(BaseModel):
    user_id: str
    answers: List[Dict[str, Any]]

# ---------- Utilities ----------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {e}")


def simple_topic_split(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    for l in lines:
        if (len(l.split()) <= 6 and l.endswith(":")) or l.isupper():
            if buf:
                chunks.append(" ".join(buf))
                buf = []
            chunks.append(l.rstrip(":"))
        else:
            buf.append(l)
    if buf:
        chunks.append(" ".join(buf))
    if len(chunks) > 12:
        step = max(1, len(chunks)//12)
        merged: List[str] = []
        temp: List[str] = []
        for i, t in enumerate(chunks):
            temp.append(t)
            if (i+1) % step == 0:
                merged.append(" ".join(temp))
                temp = []
        if temp:
            merged.append(" ".join(temp))
        chunks = merged
    return [c for c in chunks if c.strip()]


def naive_rank(question: str, chunks: List[str], top_k: int = 3) -> List[int]:
    # super-light retriever: keyword overlap score
    if not chunks:
        return []
    q = set(question.lower().split())
    scores = []
    for i, c in enumerate(chunks):
        toks = set(c.lower().split())
        scores.append((i, len(q & toks)))
    scores.sort(key=lambda x: -x[1])
    return [i for i, _ in scores[:top_k]]


def llm_summarize(text: str) -> str:
    if not text:
        return "No content found to summarize."
    prompt = f"Summarize clearly in 6-8 bullet points:\n{text[:6000]}"
    if genai and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except Exception:
            pass
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.3
            )
            return resp.choices[0].message.content
        except Exception:
            pass
    # fallback
    lines = text.split("\n")[:8]
    return "\n".join(f"- {l[:160]}" for l in lines if l.strip())


def llm_flashcards(topic_text: str, k: int = 8) -> List[Dict[str, Any]]:
    prompt = (
        "Create concise Q/A flashcards from the following study notes. "
        "Return JSON array where each item has question, answer, tags (array), difficulty (easy|medium|hard).\n\n"
        f"NOTES:\n{topic_text[:6000]}"
    )
    if genai and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return json.loads(resp.text)[:k]
        except Exception:
            pass
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2
            )
            return json.loads(resp.choices[0].message.content)[:k]
        except Exception:
            pass
    return [{"question": f"Key point {i+1}?", "answer": "", "tags": [], "difficulty": "medium"} for i in range(k)]


def llm_quiz(topic_text: str, k: int = 6) -> List[Dict[str, Any]]:
    prompt = (
        "Create a multiple-choice quiz from the notes. "
        "Return JSON array with question, options (4), correct_index, explanation and difficulty.\n\n"
        f"NOTES:\n{topic_text[:6000]}"
    )
    if genai and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return json.loads(resp.text)[:k]
        except Exception:
            pass
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2
            )
            return json.loads(resp.choices[0].message.content)[:k]
        except Exception:
            pass
    return [{"question": f"Sample Q{i+1}", "options": ["A","B","C","D"], "correct_index": 0, "difficulty":"easy", "explanation":""} for i in range(k)]


def llm_answer(question: str, context: str) -> str:
    prompt = f"Answer the question using the context. Cite sections if relevant.\nQuestion: {question}\nContext: {context[:6000]}"
    if genai and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            return model.generate_content(prompt).text
        except Exception:
            pass
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2
            )
            return resp.choices[0].message.content
        except Exception:
            pass
    return "Unable to answer without LLM API access."

# ---------- Helper DB ----------

def oid(id_str: str):
    from bson import ObjectId
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(400, "Invalid id")

# ---------- Endpoints ----------

@app.get("/")
async def root():
    return {"app": "VectorTutor API", "status": "ok"}

@app.post("/api/material/upload", response_model=UploadResponse)
async def upload_material(user_id: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    if file.filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(content)
        mat_type = "pdf"
    else:
        try:
            text = content.decode("utf-8")
            mat_type = "text"
        except Exception:
            raise HTTPException(400, "Unsupported file format")

    chunks = simple_topic_split(text)
    topics_labels = [f"Topic {i+1}" for i in range(len(chunks))]
    mat = Material(user_id=user_id, title=file.filename, type=mat_type, text=text, topics=topics_labels)
    mat_id = create_document("material", mat)

    db["material_vectors"].insert_one({"material_id": mat_id, "chunks": chunks if chunks else [text]})

    return {"material_id": mat_id, "topics": topics_labels}

@app.post("/api/material/text", response_model=UploadResponse)
async def upload_text(user_id: str = Form(...), text: str = Form(...), title: str = Form("Notes")):
    chunks = simple_topic_split(text)
    topics_labels = [f"Topic {i+1}" for i in range(len(chunks))]
    mat = Material(user_id=user_id, title=title, type="text", text=text, topics=topics_labels)
    mat_id = create_document("material", mat)
    db["material_vectors"].insert_one({"material_id": mat_id, "chunks": chunks if chunks else [text]})
    return {"material_id": mat_id, "topics": topics_labels}

@app.get("/api/material/{material_id}/summary")
async def summarize_material(material_id: str):
    m = db["material"].find_one({"_id": oid(material_id)})
    if not m:
        raise HTTPException(404, "Material not found")
    return {"summary": llm_summarize(m.get("text", ""))}

@app.post("/api/flashcards/generate")
async def generate_flashcards(user_id: str = Form(...), material_id: str = Form(...), topic_index: int = Form(0)):
    vec_doc = db["material_vectors"].find_one({"material_id": material_id}) or db["material_vectors"].find_one(sort=[("created_at", -1)])
    chunks = vec_doc.get("chunks", []) if vec_doc else []
    topic_text = chunks[topic_index] if chunks else ""
    cards = llm_flashcards(topic_text)
    out_ids = []
    for c in cards:
        fc = Flashcard(user_id=user_id, material_id=material_id, topic=f"Topic {topic_index+1}", question=c.get("question",""), answer=c.get("answer",""), tags=c.get("tags",[]), difficulty=c.get("difficulty","medium"))
        out_ids.append(create_document("flashcard", fc))
    return {"count": len(out_ids)}

@app.get("/api/flashcards")
async def list_flashcards(material_id: Optional[str] = Query(None), user_id: Optional[str] = Query(None)):
    filt: Dict[str, Any] = {}
    if material_id:
        filt["material_id"] = material_id
    if user_id:
        filt["user_id"] = user_id
    return get_documents("flashcard", filt, limit=200)

@app.post("/api/quiz/generate")
async def generate_quiz(user_id: str = Form(...), material_id: str = Form(...), topic_index: int = Form(0)):
    vec_doc = db["material_vectors"].find_one({"material_id": material_id}) or db["material_vectors"].find_one(sort=[("created_at", -1)])
    chunks = vec_doc.get("chunks", []) if vec_doc else []
    topic_text = chunks[topic_index] if chunks else ""
    items = llm_quiz(topic_text)
    out_ids = []
    for q in items:
        qi = QuizItem(user_id=user_id, material_id=material_id, topic=f"Topic {topic_index+1}", question=q.get("question",""), options=q.get("options",[]), correct_index=q.get("correct_index",0), difficulty=q.get("difficulty","easy"), explanation=q.get("explanation"))
        out_ids.append(create_document("quizitem", qi))
    return {"count": len(out_ids)}

@app.get("/api/quiz")
async def list_quiz(material_id: Optional[str] = Query(None), user_id: Optional[str] = Query(None)):
    filt: Dict[str, Any] = {}
    if material_id:
        filt["material_id"] = material_id
    if user_id:
        filt["user_id"] = user_id
    return get_documents("quizitem", filt, limit=200)

@app.post("/api/quiz/submit")
async def submit_quiz(payload: QuizSubmit):
    correct = 0
    for ans in payload.answers:
        if int(ans.get("selected", -1)) == int(ans.get("correct_index", -2)):
            correct += 1
    acc = correct / max(1, len(payload.answers))
    perf = Performance(user_id=payload.user_id, accuracy=acc, attempts=len(payload.answers))
    create_document("performance", perf)
    focus = "reinforce weak topics" if acc < 0.7 else "advance to harder material"
    plan = Plan(user_id=payload.user_id, schedule=[{"day":"tomorrow","focus":focus}], goals=["improve accuracy" if acc < 0.7 else "maintain streak"])
    create_document("plan", plan)
    return {"accuracy": acc}

@app.get("/api/performance")
async def get_performance(user_id: str):
    doc = db["performance"].find_one({"user_id": user_id}, sort=[("created_at", -1)])
    return doc or {"user_id": user_id, "accuracy": 0, "attempts": 0, "streak": 0}

@app.get("/api/plan")
async def get_plan(user_id: str):
    doc = db["plan"].find_one({"user_id": user_id}, sort=[("created_at", -1)])
    return doc or {"user_id": user_id, "schedule": [], "goals": []}

@app.get("/api/reminders")
async def get_reminders(user_id: str):
    perf = db["performance"].find_one({"user_id": user_id}, sort=[("created_at", -1)]) or {}
    streak = perf.get("streak", 0)
    return {
        "streak": streak,
        "upcoming": ["Quiz tomorrow", "Revise Topic 2"],
        "daily_goal": "Study 25 minutes"
    }

@app.post("/api/chat/ask")
async def ask_chat(req: AskRequest):
    vec_doc = db["material_vectors"].find_one({"material_id": req.material_id}) or db["material_vectors"].find_one(sort=[("created_at", -1)])
    chunks = vec_doc.get("chunks", []) if vec_doc else []
    if not chunks:
        raise HTTPException(404, "No material indexed yet")
    top_idx = naive_rank(req.question, chunks, top_k=3)
    context = "\n\n".join([chunks[i] for i in top_idx])
    answer = llm_answer(req.question, context)
    create_document("chathistory", ChatHistory(user_id=req.user_id, material_id=req.material_id, question=req.question, answer=answer, refs=[f"chunk:{int(i)}" for i in top_idx]))
    return {"answer": answer, "refs": [int(i) for i in top_idx]}

@app.get("/test")
async def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected & Working"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
