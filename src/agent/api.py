"""

FastAPI wrapper around the wafer-map agent.


    PYTHONPATH=src python3 -m agent.api
    uvicorn agent.api:app --app-dir src --reload --port 8000
"""

import os
import uuid


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent.classifier as classifier
import agent.wafer_agent as wafer_agent

_UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")

app = FastAPI(
    title="WM-811K wafer defect agent",
    description="Classify a wafer map and explain the result.",
    version="0.0.1",
)


class ClassifyRequest(BaseModel):
    wafer_map: str = Field(
        ...,
        description="path to map to be analyzed"
    )


class AnalyzeRequest(ClassifyRequest):
    session_id: str | None = Field(
        None, description="Reuse an existing session."
    )
    question: str | None = Field(
        None, description="Optional extra instruction ."
    )


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session id returned by /analyze.")
    question: str = Field(..., description="Follow-up question.")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(_UI)



@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint": classifier._CKPT,
        "checkpoint_present": os.path.exists(classifier._CKPT),
        "model": wafer_agent.MODEL,
        "llm_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/classify")
def classify(req: ClassifyRequest):
    try:
        return classifier.classify(req.wafer_map)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    session_id = req.session_id or str(uuid.uuid4())

    message = f"Analyze this wafer map: {req.wafer_map}"
    if req.question:
        message += f"\n\nAnd answer this question: {req.question}"

    try:
        prediction = classifier.classify(req.wafer_map)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        answer = wafer_agent.ask(message, session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"agent failed: {exc}") from exc

    return {"session_id": session_id, "prediction": prediction, "explanation": answer}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return {
            "session_id": req.session_id,
            "answer": wafer_agent.ask(req.question, req.session_id),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"agent failed: {exc}") from exc


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Serve the wafer-map agent API.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)
