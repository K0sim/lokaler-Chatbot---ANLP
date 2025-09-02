from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=3):
        # Erzeuge Paare: (query, document)
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)

        # Sortieren nach Score
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:top_k]]
        return top_docs
