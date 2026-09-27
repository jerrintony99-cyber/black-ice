import uuid
import requests
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database import init_db, DB_FILE
from core import ingest_document, collection, trace_impact_on_document_change

app = FastAPI(title="Sovereign Black Ice API")

init_db()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # or phi3, mistral

class IngestRequest(BaseModel):
    doc_id: str
    filename: str
    content: str

class QueryRequest(BaseModel):
    query: str

@app.post("/ingest")
def ingest(data: IngestRequest):
    doc_hash = ingest_document(data.doc_id, data.filename, data.content)
    return {"message": "Document ingested successfully", "doc_id": data.doc_id, "hash": doc_hash}

@app.post("/query")
def query_rag(data: QueryRequest):
    # Vector Search
    results = collection.query(query_texts=[data.query], n_results=2)
    
    retrieved_docs = results['documents'][0] if results['documents'] else []
    retrieved_ids = results['ids'][0] if results['ids'] else []
    
    context_str = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context_str}\n\nQuestion: {data.query}\nAnswer concise:"
    
    # Generate using Local Ollama
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })
        llm_response = response.json().get("response", "Error getting response")
    except Exception as e:
        llm_response = f"Simulated Response (Ollama connection offline): {data.query}"

    # Log query and provenance mapping
    response_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO ai_output_audit (response_id, query, generated_response, status, timestamp)
        VALUES (?, ?, ?, 'VALID', ?)
    ''', (response_id, data.query, llm_response, datetime.utcnow()))
    
    for chunk_id in retrieved_ids:
        cursor.execute('''
            INSERT INTO response_provenance (response_id, chunk_id)
            VALUES (?, ?)
        ''', (response_id, chunk_id))
        
    conn.commit()
    conn.close()

    return {
        "response_id": response_id,
        "query": data.query,
        "answer": llm_response,
        "source_chunk_ids": retrieved_ids
    }

@app.post("/update-document")
def update_doc(data: IngestRequest):
    impact = trace_impact_on_document_change(data.doc_id, data.content)
    return impact

@app.get("/audit-logs")
def get_audit_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT response_id, query, generated_response, status, timestamp FROM ai_output_audit")
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "response_id": r[0],
            "query": r[1],
            "response": r[2],
            "status": r[3],
            "timestamp": r[4]
        }
        for r in rows
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)