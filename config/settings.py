# === Lokale Vektordatenbank ===
DB_PATH = "./chromadb_store"
COLLECTION_NAME = "regelwerk"

# === Embedding-Modell für SentenceTransformer ===
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# === Ollama LLM-Konfiguration ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  

# === Chunking-Strategie ===
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50

# === K für Top-K-Ähnlichkeitsabfrage ===
TOP_K_RESULTS = 3
