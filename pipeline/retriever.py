from chromadb import PersistentClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from config.settings import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from pipeline.reranker import ReRanker


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
        n_results=top_k * 2,  # hole mehr für besseres Re-Ranking
        include=["documents", "metadatas", "distances"]
    )

    emb_docs = emb_results["documents"][0]
    emb_meta = emb_results["metadatas"][0]
    emb_dists = emb_results["distances"][0]
    emb_scores = [1 - d for d in emb_dists]  # cosine distance -> similarity

    # BM25 Scores
    bm25_scores = self.bm25.get_scores(query.split())

    # Kombinieren (optional normalisieren)
    hybrid_scores = []
    for i, doc in enumerate(emb_docs):
        try:
            idx = self.documents.index(doc)
        except ValueError:
            continue  # Falls das Dokument nicht gefunden wird (sollte selten sein)
        bm_score = bm25_scores[idx]
        emb_score = emb_scores[i]
        score = alpha * bm_score + (1 - alpha) * emb_score
        hybrid_scores.append((score, doc, emb_meta[i]))

    # Sortieren (noch nicht final: das macht ReRanker)
    hybrid_sorted = sorted(hybrid_scores, key=lambda x: x[0], reverse=True)
    docs_for_rerank = [t[1] for t in hybrid_sorted]
    metas_for_rerank = [t[2] for t in hybrid_sorted]

    # === NEU: Re-Ranking lokal ===
    reranker = ReRanker()
    reranked_docs = reranker.rerank(query, docs_for_rerank, top_k=top_k)

    # Hole die Metadaten passend zu den rerankten docs
    top_chunks = [(0, doc, metas_for_rerank[docs_for_rerank.index(doc)]) for doc in reranked_docs]

    # Bisherige Rückgabevariablen beibehalten
    contexts = [t[1] for t in top_chunks]
    sources = [f"{m['source']} (S. {m['page']}, Titel: {m['title']})" for _, _, m in top_chunks]

    return contexts, [m for _, _, m in top_chunks]

