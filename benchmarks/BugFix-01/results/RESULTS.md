# BugFix-01 — Incomplete `/delete` in Hybrid RAG

**Date:** 2026-07-18  
**Project:** `hybrid_rag` (`main.py`)  
**Method:** Same prompt → 4 local Ollama coding models → scored after the run  
**Prompt:** [`prompt/prompt.txt`](../prompt/prompt.txt)  
**Runner:** [`run_benchmark.py`](../run_benchmark.py)

---

## The real bug (verified in code)

`GET /delete` only clears the in-memory `documents` list.

It does **not**:

1. Delete rows from Postgres `document_chunks`
2. Reset `bm25_index` / `faiss_index`
3. Remove `vector_index.faiss`

On restart, `lifespan` → `rebuild_indexes()` reloads everything from the DB (and can keep a stale FAISS file). Documents “come back.”

**Correct fix must do all of:**

- `DELETE` all `DocumentChunk` rows (commit)
- Clear / reset in-memory `documents`, `bm25_index`, `faiss_index`
- Remove `INDEX_FILE` (or otherwise prevent stale FAISS reload)
- Capture cleared count **before** clearing the list

---

## Models tested

| # | Model | Size (approx) |
|---|-------|---------------|
| 1 | `qwen3.6:35b-a3b-coding-nvfp4` | 21 GB |
| 2 | `qwen3-coder-fast:30b` | 18 GB |
| 3 | `north-mini-code-1.0:mlx-nvfp4` | 19 GB |
| 4 | `glm-4.7-flash:latest` | 19 GB |

Temperature: `0.1` via `ollama.chat`.

---

## Scoring rubric

| Criteria | Weight |
| -------- | -----: |
| Correctness | 40% |
| Root cause | 20% |
| Code quality | 15% |
| Explanation | 10% |
| Performance (relative) | 10% |
| Extra improvements | 5% |

---

## Run results (timed)

| Model | Time | Correct fix? | Notes |
| ----- | ---: | ------------ | ----- |
| qwen3.6:35b-a3b-coding-nvfp4 | 80.4s | ✅ | Full fix: DB + memory + indexes + FAISS file |
| qwen3-coder-fast:30b | 12.26s | ⚠️ Partial | Same structure, but returns `len(documents)` **after** clear → always `0` |
| north-mini-code-1.0:mlx-nvfp4 | 80.37s | ✅ | Full fix; best insight on empty-DB `rebuild_indexes` early return |
| glm-4.7-flash:latest | 124.87s | ❌ | DB delete yes; misses FAISS file delete; `bm25_index = None` without `global` |

Raw timing JSON: [`raw_run_summary.json`](raw_run_summary.json)

---

## Detailed scores

### 1) `qwen3.6:35b-a3b-coding-nvfp4` — **94 / 100** (winner)

| Criteria | Score | Why |
| -------- | ----: | --- |
| Correctness | 40/40 | Deletes DB rows, clears memory, nulls indexes, removes FAISS file, counts before clear |
| Root cause | 20/20 | Exact: memory-only clear → DB survives → restart rebuild |
| Code quality | 14/15 | Clean, ordered, uses `global` correctly |
| Explanation | 10/10 | Concise and accurate |
| Performance | 5/10 | 80.4s (mid pack) |
| Extra | 5/5 | Search-guard note, O(N) rebuild cost, index consistency |

**Response:** [`responses/qwen3_6_35b-a3b-coding-nvfp4.md`](../responses/qwen3_6_35b-a3b-coding-nvfp4.md)

---

### 2) `north-mini-code-1.0:mlx-nvfp4` — **90 / 100**

| Criteria | Score | Why |
| -------- | ----: | --- |
| Correctness | 40/40 | Same complete fix as Qwen3.6 Coding |
| Root cause | 18/20 | Strong on Postgres + FAISS file; slightly overstated current code as already clearing BM25/FAISS (it does not) |
| Code quality | 13/15 | Correct logic; a bit noisier (extra import comments) |
| Explanation | 9/10 | Clear |
| Performance | 5/10 | 80.37s |
| Extra | 5/5 | Excellent: empty DB makes `rebuild_indexes()` return early and leave stale globals |

**Response:** [`responses/north-mini-code-1_0_mlx-nvfp4.md`](../responses/north-mini-code-1_0_mlx-nvfp4.md)

---

### 3) `qwen3-coder-fast:30b` — **78 / 100**

| Criteria | Score | Why |
| -------- | ----: | --- |
| Correctness | 28/40 | Deletes DB + file + indexes, but **return count is always 0** after `documents.clear()` |
| Root cause | 20/20 | Correct |
| Code quality | 10/15 | Introduced a new bug in the fix; contradictory “call rebuild_indexes” advice |
| Explanation | 8/10 | Fine, shorter |
| Performance | 10/10 | Fastest by far (12.26s) |
| Extra | 2/5 | Partial-deletion note OK; rebuild advice misguided |

**Response:** [`responses/qwen3-coder-fast_30b.md`](../responses/qwen3-coder-fast_30b.md)

---

### 4) `glm-4.7-flash:latest` — **55 / 100**

| Criteria | Score | Why |
| -------- | ----: | --- |
| Correctness | 20/40 | Deletes DB, but does **not** remove `INDEX_FILE`; `rebuild_indexes()` no-ops on empty DB; `bm25_index = None` without `global` is a local assignment |
| Root cause | 16/20 | Got Postgres part; underplayed on-disk FAISS persistence |
| Code quality | 6/15 | Global scoping bug; misleading rebuild call |
| Explanation | 8/10 | Readable |
| Performance | 2/10 | Slowest (124.87s) |
| Extra | 3/5 | Some architecture comments; less actionable |

**Response:** [`responses/glm-4_7-flash_latest.md`](../responses/glm-4_7-flash_latest.md)

---

## Leaderboard

| Rank | Model | Score | Time |
| ---: | ----- | ----: | ---: |
| 1 | qwen3.6:35b-a3b-coding-nvfp4 | **94** | 80.4s |
| 2 | north-mini-code-1.0:mlx-nvfp4 | **90** | 80.37s |
| 3 | qwen3-coder-fast:30b | **78** | 12.26s |
| 4 | glm-4.7-flash:latest | **55** | 124.87s |

---

## Headline candidates (data-driven)

Winner was Qwen3.6 Coding — expected coding specialist did win, but North Mini was nearly tied on quality, and the fastest model shipped a broken return value.

Suggested titles:

1. **The Model I Expected to Win Actually Did. Here's the Data.**
2. **I Gave 4 Local Coding Models the Same FastAPI Bug. The Fastest One Broke Its Own Fix.**
3. **Qwen3.6 Coding Beat North Mini by 4 Points on a Real Hybrid RAG Delete Bug.**

---

## Reference fix (evaluator ground truth)

```python
@app.get("/delete")
async def delete_docs():
    global documents, bm25_index, faiss_index

    cleared_count = len(documents)

    db = SessionLocal()
    try:
        db.query(DocumentChunk).delete()
        db.commit()
    finally:
        db.close()

    documents.clear()
    bm25_index = None
    faiss_index = None

    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)

    return {"total cleared docs": cleared_count}
```

Note: HTTP method should ideally be `DELETE`, not `GET` — out of scope for this bug’s core failure mode, but a fair related improvement.

---

## Reproducibility

```bash
cd /Applications/htdocs/AgenticAITutorials/hybrid_rag
venv/bin/python benchmarks/BugFix-01/run_benchmark.py
```

Then re-score responses under `responses/` with the same rubric. Do not change the prompt between runs.
