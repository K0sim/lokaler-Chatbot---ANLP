from chromadb import PersistentClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from config.settings import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

class HybridRetriever:
    def __init__(self, db_path=DB_PATH, collection_name=COLLECTION_NAME, embedding_model_name=EMBEDDING_MODEL_NAME):
        self.client = PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Chunks für BM25 vorbereiten
        results = self.collection.get(include=["documents"])
        self.documents = results["documents"]
        self.ids = results["ids"]
        self.metadatas = results["metadatas"]

        self.bm25 = BM25Okapi([doc.split() for doc in self.documents])

    def retrieve_context(self, query: str, top_k: int = 5, alpha: float = 0.5):
        query_embedding = self.embedding_model.encode(query).tolist()

        # Embedding-Scores
        emb_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"]
        )

        emb_docs = emb_results["documents"][0]
        emb_meta = emb_results["metadatas"][0]
        emb_dists = emb_results["distances"][0]
        emb_scores = [1 - d for d in emb_dists]  # cosine distance -> similarity

        # BM25 Scores
        bm25_scores = self.bm25.get_scores(query.split())
        # bm25_ranked = np.argsort(bm25_scores)[::-1][:top_k * 2]

        # Kombinieren (optional normalisieren)
        hybrid_scores = []
        for i, doc in enumerate(emb_docs):
            idx = self.documents.index(doc)
            bm_score = bm25_scores[idx]
            emb_score = emb_scores[i]
            score = alpha * bm_score + (1 - alpha) * emb_score
            hybrid_scores.append((score, doc, emb_meta[i]))

        # Sortieren & Rückgabe
        top_chunks = sorted(hybrid_scores, key=lambda x: x[0], reverse=True)[:top_k]
        contexts = [t[1] for t in top_chunks]
        sources = [f"{m['source']} (S. {m['page']}, Titel: {m['title']})" for _, _, m in top_chunks]

        return contexts, [m for _, _, m in top_chunks]
