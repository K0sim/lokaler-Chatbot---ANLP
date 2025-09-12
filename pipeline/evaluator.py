# pipeline/evaluator.py

import requests
from config.settings import OLLAMA_URL, MODEL_NAME


class LLMJudge:
    def __init__(self, ollama_url=OLLAMA_URL, model_name=MODEL_NAME):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def evaluate(self, frage: str, antwort: str, kontext: str, referenz: str = None) -> str:
        """
        Bewertet die Antwort eines LLM auf eine Frage anhand des gegebenen Kontexts.
        Optional: Vergleich mit einer bekannten Referenzantwort.
        """
        prompt = self._build_prompt(frage, antwort, kontext, referenz)
        print("🧑‍⚖️ Bewertung wird erstellt…")

        try:
            response = requests.post(
                url=self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()

        except requests.RequestException as e:
            print(f"❌ Bewertungsfehler: {e}")
            return "Fehler bei der Bewertung durch das Bewertungssystem."

    def _build_prompt(self, frage: str, antwort: str, kontext: str, referenz: str = None) -> str:
        base_prompt = f"""
Du bist ein **strenger, neutraler Prüfer**, der ein Retrieval-Augmented Generation (RAG)-System evaluiert.

Bewerte die Antwort auf Grundlage des bereitgestellten Kontexts. Dabei gelten folgende strenge Kriterien – jede Bewertung erfolgt auf einer Skala von 1 bis 5 (1 = sehr schlecht, 5 = ausgezeichnet):

🔶 Relevanz: Passt der gefundene Kontext zur Frage?
🔶 Korrektheit: Ist die Antwort **faktisch korrekt** und **direkt durch den Kontext belegbar**?
🔶 Vollständigkeit: Beantwortet die Antwort **alle Aspekte der Frage** vollständig?
🔶 Kontexttreue: Hält sich die Antwort **streng** an den Kontext, ohne zusätzliche Informationen zu erfinden?

⚠️ Hinweise:
- Wenn eine Aussage nicht im Kontext nachgewiesen werden kann → **Punktabzug bei Korrektheit und Kontexttreue**
- Auch plausible Aussagen zählen als **Halluzination**, wenn sie nicht im Kontext stehen.
- Wenn die Antwort nur einen Teil der Frage behandelt → **Punktabzug bei Vollständigkeit**

Gib für jedes Kriterium eine **knappe Begründung**.

Beende mit:
- einer **Gesamtnote (1–5)**,
- einem **Verbesserungsvorschlag** (wenn möglich).

---

🟡 Frage:
{frage}

🟢 Antwort:
{antwort}

📄 Kontext (aus Dokumenten):
{kontext}
"""

        if referenz:
            base_prompt += f"""

✅ Referenzantwort (optional):
{referenz}
"""

        base_prompt += "\n\n📈 Jetzt bitte bewerten:"
        return base_prompt.strip()
