import requests
from config.settings import OLLAMA_URL, MODEL_NAME


class AnswerGenerator:
    def __init__(self, ollama_url=OLLAMA_URL, model_name=MODEL_NAME):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def generate_answer(self, context: str, question: str) -> str:
        """
        Baut einen Prompt mit Kontext + Frage und sendet ihn an das lokale LLM via Ollama.
        """
        prompt = f"Kontext:\n{context}\n\nFrage: {question}\nAntwort:"
        print("Sende Anfrage an Ollama...")

        try:
            response = requests.post(
                url=self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 250
                                }                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()

        except requests.RequestException as e:
            print(f"❌ Fehler beim Abrufen von Ollama: {e}")
            return "Entschuldigung, es gab ein Problem beim Abrufen der Antwort vom Sprachmodell."
