import re
from typing import List

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

def split_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_title(line: str) -> str | None:
    """
    Erkennt Titel anhand typischer Formatierung:
    - optional nummeriert (z. B. 5.1.2)
    - beginnt mit Großbuchstaben
    - erlaubt bestimmte Satzzeichen
    """
    line = line.strip()
    title_regex = re.compile(
        r"^(?:\d+(\.\d+)*\s+)?[A-ZÄÖÜ][\wäöüÄÖÜß\s,.!?:;\"'()/-]{2,100}$"
    )
    if title_regex.match(line):
        return line
    return None


def is_probably_table(text: str) -> bool:
    """
    Heuristik zur Tabellenerkennung: viele Tabs oder Leerzeichenblöcke
    """
    lines = text.split("\n")
    tab_lines = sum(1 for line in lines if "\t" in line or re.search(r"\s{2,}", line))
    return tab_lines > 3

