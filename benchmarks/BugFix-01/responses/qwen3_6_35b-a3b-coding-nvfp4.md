# Model: `qwen3.6:35b-a3b-coding-nvfp4`

- Elapsed: **80.4s**
- Error: `None`

## Response

## Root Cause
The `delete_docs` endpoint only clears the in-memory `documents` list but never removes records from the Postgres `document_chunks` table. Upon server restart, the `lifespan` hook calls `rebuild_indexes()`, which queries the database and repopulates the in-memory state with the previously "deleted" data.

## Fixed Code
```python
@app.get("/delete")
async def delete_docs():
    global documents, bm25_index, faiss_index
    
    cleared_count = len(documents)
    
    # 1. Permanently remove records from Postgres
    db = SessionLocal()
    try:
        db.query(DocumentChunk).delete()
        db.commit()
    finally:
        db.close()
        
    # 2. Reset in-memory structures
    documents.clear()
    bm25_index = None
    faiss_index = None
    
    # 3. Remove stale FAISS index file to prevent reload on restart
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        
    return {"total cleared docs": cleared_count}
```

## Related Bugs & Improvements
- **Search endpoint crash risk:** Setting `bm25_index` and `faiss_index` to `None` will crash any search call before a new upload. Add a guard in your search logic or call `rebuild_indexes()` after clearing to ensure valid empty indexes.
- **Inefficient rebuilds:** `rebuild_indexes()` runs on every `POST /upload`, making it O(N). Consider incremental index updates or a background task for large datasets.
- **Index file consistency:** The `lifespan` hook reads the FAISS file before calling `rebuild_indexes()`. Removing the file on delete prevents stale reloads, but ensure `rebuild_indexes()` remains the single source of truth for index persistence.
