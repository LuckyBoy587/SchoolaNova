from fastapi import FastAPI, HTTPException
from model import SimpleRag  # Assumes your class is in this file

# --- FastAPI App and Startup Event ---
app = FastAPI()
rag_instance = None
MODEL_FILE = "./book.txt"  # <-- CHANGE THIS

@app.on_event("startup")
def startup_event():
    global rag_instance
    try:
        rag_instance = SimpleRag(fileName=MODEL_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Startup failed: {e}")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "RAG API is running. Check /docs for endpoints."}

@app.post("/get_answer")
def get_answer_endpoint(query: str, k: int = 3):
    """Retrieves and rephrases an answer for a query."""
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")
    
    answer = rag_instance.get_answer(query=query, k=k)
    if not answer:
        raise HTTPException(status_code=404, detail="No relevant answer found.")
    
    return {"query": query, "answer": answer}

@app.post("/get_questions")
def get_questions_endpoint(chapter: str, n: int = 3):
    """Generates questions on a given topic."""
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    questions = rag_instance.get_questions(chapter=chapter, n=n)
    if not questions:
        raise HTTPException(status_code=404, detail="Could not generate questions.")
    
    return {"chapter": chapter, "questions": questions}
