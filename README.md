
# Lokaler RAG-Chatbot mit PyQt6, ChromaDB und Ollama

Dieses Projekt ist ein lokaler Retrieval-Augmented-Generation (RAG) Chatbot mit grafischer Benutzeroberfläche. Er nutzt SentenceTransformers, ChromaDB und ein lokales Sprachmodell über Ollama zur Beantwortung technischer Fragen auf Basis eigener PDF-Dokumente.

---

## Voraussetzungen

- Windows oder Linux
- Python 3.10 oder 3.11
- [Ollama](https://ollama.com/download) (lokaler LLM-Dienst, z. B. Mistral)
- Git (optional)

---

## Einrichtung

### 1. Repository klonen oder Dateien bereitstellen

```bash
git clone <REPO-URL>
cd lokal_chatbot
```

Oder Projektordner manuell entpacken.

### 2. Virtuelle Umgebung erstellen und aktivieren

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Abhängigkeiten installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Lokales Sprachmodell mit Ollama starten

Falls noch nicht geschehen, lade und starte das Modell (z. B. Mistral):

```bash
ollama run mistral
```

Das Modell muss im Hintergrund laufen, während du den Chatbot nutzt.

---

## Datenbank vorbereiten

Lege deine PDF-Dateien in folgenden Ordner:

```
data/pdfs/
```

Dann führe aus:

```bash
python setup_chromadb.py
```

Dies lädt die Dateien, chunked den Text und speichert Vektoren in ChromaDB.

---

## Chatbot starten

```bash
python UI/app_desktop.py
```

Es öffnet sich ein Fenster mit Texteingabe, Antwortausgabe, Quellenangabe und automatischer Antwortbewertung.

---

## Komponenten

- `UI/app_desktop.py` – Hauptanwendung mit Benutzeroberfläche
- `pipeline/retriever.py` – Hybrid-Retriever (BM25 + Embedding)
- `pipeline/generator.py` – Antwortgenerierung via Ollama
- `pipeline/evaluator.py` – Bewertung der Antwort mit LLM-as-a-Judge
- `setup_chroma.py` – Einmaliges Setup zur Indexierung von PDFs
- `config/settings.py` – Pfade, Modellnamen, DB-Konfiguration

---

## Konfiguration

Alle zentralen Parameter sind in `config/settings.py` definiert:

- Pfade (`DB_PATH`, `PDF_DIR`)
- Collection-Name in ChromaDB
- Embedding-Modell
- LLM-Modell für Antworten und Bewertung
- Ollama-Endpunkt

---

## Tipps zur Nutzung

- Ändere `CHUNK_SIZE`, `CHUNK_OVERLAP` oder `alpha` im Retriever für bessere Ergebnisse
- Nutze verschiedene Ollama-Modelle über `MODEL_NAME`
- Bewerte Qualität, Relevanz und Kontexttreue automatisch oder manuell

---

## Deinstallation

```bash
deactivate
rmdir /s /q .venv  # Windows
# rm -rf .venv  # Linux/macOS
```

---

## Hinweise

- Die Bewertung erfolgt lokal über dasselbe LLM wie die Antwortgenerierung (z. B. Mistral)
- Die automatische Bewertung ersetzt keine manuelle Prüfung, sondern dient als Anhaltspunkt

---

## Lizenz

Dieses Projekt ist ausschließlich für Forschungs- und Ausbildungszwecke vorgesehen.
