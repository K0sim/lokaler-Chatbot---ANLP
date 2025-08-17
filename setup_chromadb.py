import os
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from utils.text_splitter import extract_title, split_text, is_probably_table

# Konfiguration
PDF_DIR = "./chromadb_store/pdfs"
DB_DIR = "./chromadb_store"
COLLECTION_NAME = "regelwerk"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE=16 # Für schnellere Verarbeitung 


# 1. Embedding-Modell laden
print("Lade Embedding-Modell...")
model = SentenceTransformer(EMBEDDING_MODEL)

# 2. ChromaDB-Client
print("Initialisiere ChromaDB Client...")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def load_pdfs(folder):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            docs.append((filename, path))
    return docs


# 3. Funktionen
def chunk_with_metadata(filename, pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    metadatas = []

    for page_num, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if not raw_text:
            continue

        lines = raw_text.splitlines()
        buffer = []
        current_title = "Unbekannt"

        for line in lines:
            title_candidate = extract_title(line)
            if title_candidate:
                if buffer:
                    joined = "\n".join(buffer).strip()
                    for chunk in split_text(joined):
                        chunks.append(chunk)
                        metadatas.append({
                            "source": filename,
                            "page": page_num + 1,
                            "title": current_title,
                            "is_table": is_probably_table(chunk)
                        })
                    buffer = []
                current_title = title_candidate
                continue
            buffer.append(line)

        if buffer:
            joined = "\n".join(buffer).strip()
            for chunk in split_text(joined):
                chunks.append(chunk)
                metadatas.append({
                    "source": filename,
                    "page": page_num + 1,
                    "title": current_title,
                    "is_table": is_probably_table(chunk)
                })

    return chunks, metadatas



# 6. Verarbeitung starten
all_docs = load_pdfs(PDF_DIR)
doc_id = 0

if not all_docs:
    print("⚠️ Keine PDF-Dateien gefunden. Bitte lege Dateien in:", PDF_DIR)
else:
    print(f"📂 {len(all_docs)} PDF(s) gefunden. Starte Einbettung...\n")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for filename, path in tqdm(all_docs, desc="📄 PDFs analysieren"):
        chunks, metadatas = chunk_with_metadata(filename, path)
        all_chunks.extend(chunks)
        all_metadatas.extend(metadatas)
        all_ids.extend([str(doc_id + i) for i in range(len(chunks))])
        doc_id += len(chunks)

    # 7. Embedding in Batches
    print(f"🛠 Embedde {len(all_chunks)} Chunks in Batches (Größe: {BATCH_SIZE})...\n")
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="🔄 Chunks einbetten"):
        batch_chunks = all_chunks[i:i+BATCH_SIZE]
        batch_metadatas = all_metadatas[i:i+BATCH_SIZE]
        batch_ids = all_ids[i:i+BATCH_SIZE]

        batch_embeddings = model.encode(batch_chunks, show_progress_bar=False).tolist()
        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadatas,
            ids=batch_ids,
            embeddings=batch_embeddings,
        )

    print(f"\n✅ Fertig: {doc_id} Abschnitte wurden indexiert und gespeichert unter: {DB_DIR}")
