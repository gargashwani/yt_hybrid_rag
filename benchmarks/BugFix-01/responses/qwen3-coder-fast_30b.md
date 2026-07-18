# Model: `qwen3-coder-fast:30b`

- Elapsed: **12.26s**
- Error: `None`

## Response

### Root Cause

The bug occurs because the `/delete` endpoint only clears the in-memory `documents` list but doesn't remove the corresponding records from the Postgres database. When the server restarts, `rebuild_indexes()` re-reads all chunks from the database (including the deleted ones), restoring the search index with the old data.

### Fixed Code

```python
@app.get("/delete")
async def delete_docs():
    global faiss_index, bm25_index, documents
    db = SessionLocal()
    try:
        # Delete all records from database
        db.query(DocumentChunk).delete()
        db.commit()
    finally:
        db.close()
    
    # Clear in-memory state
    documents.clear()
    faiss_index = None
    bm25_index = None
    
    # Remove index file to prevent reuse
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        
    return {
        "total cleared docs": len(documents)
    }
```

### Related Issues

1. The `rebuild_indexes()` function should be called after deletion to properly reset the in-memory state.
2. Consider adding a check in `lifespan` to handle case where index file exists but DB is empty.
3. The current implementation doesn't support partial deletion (only full clear).
