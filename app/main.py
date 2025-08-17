from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.retriever import HybridRetriever
from pipeline.generator import AnswerGenerator
from config.settings import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

app = FastAPI(title="Lokaler RAG Chatbot")

# === Klassen ===
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class AnswerResponse(BaseModel):
    answer: str
    context: str
    sources: list

# === Komponenten initialisieren ===
HybridRetriever = HybridRetriever(DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME)
generator = AnswerGenerator()

# === Endpunkte ===
@app.get("/")
def read_root():
    return {"message": "Chatbot ist bereit. Sende eine POST-Anfrage an /query"}

@app.post("/query", response_model=AnswerResponse)
def query_answer(req: QueryRequest):
    print(f"📥 Frage empfangen: {req.question}")

    # 1. Kontext + Metadaten abrufen
    contexts, metadaten = HybridRetriever.retrieve_context(req.question, top_k=req.top_k)

    # 2. Kontext mit Quellen strukturieren
    context_string = ""
    sources = []
    for i, (chunk, meta) in enumerate(zip(contexts, metadaten)):
        source_str = f"Quelle {i+1} | {meta['source']} | S. {meta['page']} | {meta['title']}"
        context_string += f"\n[{source_str}]\n{chunk}\n"
        sources.append(source_str)

    # 3. Antwort generieren
    answer = generator.generate_answer(context_string, req.question)

    return {
        "answer": answer,
        "context": context_string.strip(),
        "sources": sources
    }
