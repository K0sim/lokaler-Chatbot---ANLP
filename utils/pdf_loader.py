import os
from typing import List, Tuple
from pypdf import PdfReader

def load_pdfs(folder_path: str) -> List[Tuple[str, str]]:
    """
    Listet alle PDF-Dateien in einem Verzeichnis auf.
    Rückgabe: Liste von (Dateiname, Pfad)-Tupeln
    """
    pdfs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            pdfs.append((file, full_path))
    return pdfs

def extract_text_by_page(pdf_path: str) -> List[str]:
    """
    Extrahiert Text seitenweise aus einer PDF-Datei.
    Rückgabe: Liste von Strings pro Seite
    """
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            texts.append(text if text else "")
        except Exception as e:
            texts.append("")  # Fehlerhafte Seite ignorieren
    return texts
