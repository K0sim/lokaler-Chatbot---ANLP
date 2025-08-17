from chromadb import PersistentClient

client = PersistentClient(path="./chromadb_store")
collection = client.get_collection(name="regelwerk")

peeked = collection.peek(5)

for i, doc in enumerate(peeked["documents"]):
    meta = peeked["metadatas"][i]
    print(f"\n🔹 Chunk {i+1}")
    print(f"📄 Quelle: {meta['source']}, Seite {meta['page']}, Titel: {meta['title']}, Tabelle: {meta['is_table']}")
    print(f"📝 Textauszug:\n{doc[:500]}...")
