"""
Tahqiq.ai — knowledge_base.py
ChromaDB-backed HEC University Knowledge Base

Architecture
────────────
  ┌──────────────────────────────────────────────────────────┐
  │  HEC University Records (UniversityRecord)               │
  │         │                                                 │
  │         ▼                                                 │
  │  Document Builder  →  rich natural-language "document"   │
  │    + Urdish synonym layer (Roman Urdu keyword hints)      │
  │         │                                                 │
  │         ▼                                                 │
  │  Embedding Strategy (auto-selected at startup)           │
  │    ① SentenceTransformer  — best quality, needs network  │
  │       "all-MiniLM-L6-v2"  (384-dim, ~90 MB)              │
  │    ② TF-IDF + cosine      — zero network, always works   │
  │         │                                                 │
  │         ▼                                                 │
  │  ChromaDB Collection  (persistent on disk)               │
  │    • vector index  (cosine HNSW)                         │
  │    • metadata store (city, type, tags, fee, merit …)     │
  │         │                                                 │
  │         ▼                                                 │
  │  semantic_search(query, intent, top_k)                   │
  │    → hybrid: embedding similarity + metadata $where      │
  └──────────────────────────────────────────────────────────┘

Public API
──────────
  get_collection()                    → initialised ChromaDB collection
  add_universities(records)           → index a list of UniversityRecord
  semantic_search(query, intent, k)   → list[UniversityRecord]  (top-k)
  get_collection_stats()              → dict with counts / health info
  reset_collection()                  → wipe and re-seed (dev helper)
  seed_if_empty()                     → idempotent startup seeder
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
from chromadb.config import Settings

# Re-use shared models — no circular imports
from agent_logic import UniversityRecord, StudentIntent, _HEC_KB  # _HEC_KB is CSV-seeded (209 universities)

logger = logging.getLogger("tahqiq.kb")
logging.basicConfig(level=logging.INFO)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_PATH          = Path(os.environ.get("CHROMA_PATH", "./chroma_db"))
COLLECTION_NAME      = "hec_universities"
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# ══════════════════════════════════════════════════════════════════════════════
# Embedding Strategy — dual-mode
# ══════════════════════════════════════════════════════════════════════════════

class _EmbeddingBackend:
    """Abstract base. Subclasses must implement embed()."""
    name: str = "base"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class _SentenceTransformerBackend(_EmbeddingBackend):
    """
    Primary backend: SentenceTransformer (all-MiniLM-L6-v2).
    384-dimensional cosine embeddings. Multilingual-aware.
    Requires: pip install sentence-transformers + network on first load.
    """
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer '%s' …", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim   = self._model.get_sentence_embedding_dimension()
        logger.info("SentenceTransformer loaded (dim=%d)", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vecs.tolist()


class _TFIDFBackend(_EmbeddingBackend):
    """
    Fallback backend: TF-IDF vectorizer fitted on the HEC corpus.

    Why TF-IDF rather than random hashing?
    • Hashing collides — high-frequency domain terms (e.g. "engineering",
      "Multan") get drowned out in a small hash space.
    • TF-IDF weights rare, discriminative terms higher and normalises
      document length — perfect for a 6-200 university corpus.
    • Fitted on the full HEC document corpus, vocabulary is precisely
      aligned with what students query about (Urdish + English).
    • Zero network calls, zero model downloads, deterministic.

    Dimension: min(512, vocab_size) — typically ~400-500 for HEC corpus.
    """
    name = "tfidf"
    _CACHE_FILE = "tfidf_vectorizer.pkl"

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        cache_path = CHROMA_PATH / self._CACHE_FILE
        if cache_path.exists():
            logger.info("Loading cached TF-IDF vectorizer from %s", cache_path)
            with open(cache_path, "rb") as f:
                self._vec: TfidfVectorizer = pickle.load(f)
        else:
            logger.info("Fitting TF-IDF vectorizer on %d documents …", len(corpus))
            self._vec = TfidfVectorizer(
                max_features=512,
                ngram_range=(1, 2),       # unigrams + bigrams → "computer science"
                sublinear_tf=True,        # log(1+tf) reduces burstiness
                strip_accents="unicode",
                analyzer="word",
                min_df=1,
            )
            self._vec.fit(corpus)
            CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self._vec, f)
            logger.info("TF-IDF fitted and cached (%d features).",
                        len(self._vec.vocabulary_))

    @property
    def dim(self) -> int:
        return len(self._vec.vocabulary_)

    def embed(self, texts: list[str]) -> list[list[float]]:
        mat   = self._vec.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).tolist()


# ── Lazy singleton ──────────────────────────────────────────────────────────

_embedding_backend: Optional[_EmbeddingBackend] = None


def _get_embedding_backend(corpus: Optional[list[str]] = None) -> _EmbeddingBackend:
    """
    Auto-select embedding backend:
    ① SentenceTransformer if available (production).
    ② TF-IDF fitted on corpus (offline / CI / no network).
    """
    global _embedding_backend
    if _embedding_backend is not None:
        return _embedding_backend

    try:
        _embedding_backend = _SentenceTransformerBackend(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        logger.warning("SentenceTransformer unavailable (%s) → TF-IDF fallback.", exc)
        if corpus is None:
            corpus = [_build_document(u) for u in _HEC_KB]
        _embedding_backend = _TFIDFBackend(corpus)

    logger.info("Active embedding backend: %s", _embedding_backend.name)
    return _embedding_backend


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB EmbeddingFunction adapter
# ══════════════════════════════════════════════════════════════════════════════

class _TahqiqEmbeddingFunction(chromadb.EmbeddingFunction):
    """Wraps the active _EmbeddingBackend for ChromaDB's add() / query()."""
    def __init__(self, backend: _EmbeddingBackend):
        self._backend = backend

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return self._backend.embed(list(input))


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB singleton
# ══════════════════════════════════════════════════════════════════════════════

_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client → %s", CHROMA_PATH.resolve())
    return _chroma_client


# ══════════════════════════════════════════════════════════════════════════════
# Document builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_document(u: UniversityRecord) -> str:
    """
    Convert a UniversityRecord into a rich natural-language document.
    Includes Urdish synonyms so Roman Urdu queries surface correct results.
    Kept under ~300 tokens (MiniLM context window).
    """
    sc_text    = "; ".join(f"{s['name']} ({s['coverage']})" for s in u.scholarships) \
                 if u.scholarships else "koi scholarship nahi / no scholarship listed"
    fee_str    = (f"PKR {u.annual_fee_pkr:,} per year (sasti / affordable / low fee)"
                  if u.annual_fee_pkr and u.annual_fee_pkr <= 50_000
                  else f"PKR {u.annual_fee_pkr:,} per year" if u.annual_fee_pkr
                  else "fee information not available")
    merit_str  = f"minimum merit cutoff {u.merit_cutoff}%" if u.merit_cutoff else "merit cutoff not specified"
    emp_str    = f"{u.employment_rate}% graduates employed within 6 months" if u.employment_rate else "employment data not available"
    hostel_str = "hostel available on campus, accommodation provided" if u.has_hostel else "no hostel, day scholar only"
    fields_str = ", ".join(u.fields_offered)

    return f"""
{u.name} is a {u.type.lower()} university located in {u.city}, {u.province}, Pakistan.
HEC ranking category: {u.hec_category}. {emp_str}.
Programs offered: {fields_str}.
Annual tuition fee: {fee_str}. {merit_str}.
{hostel_str}.
Scholarships: {sc_text}.
Website: {u.website}.
{_urdish_tag_layer(u)}
""".strip()


def _urdish_tag_layer(u: UniversityRecord) -> str:
    """
    Deterministic Urdish keyword hints derived from structured data fields.
    Every tag is a logical consequence of a real data value — never hallucinated.
    """
    tags: list[str] = []

    if u.annual_fee_pkr and u.annual_fee_pkr <= 40_000:
        tags += ["sasti university", "kam fees", "affordable", "low fee", "budget friendly"]
    elif u.annual_fee_pkr and u.annual_fee_pkr <= 80_000:
        tags += ["medium fees", "moderate cost", "reasonable fee"]
    else:
        tags += ["mehngi university", "high fee", "expensive"]

    if u.scholarships:
        tags += ["scholarship milti hai", "scholarship available", "financial aid", "wazifa"]
    if u.has_hostel:
        tags += ["hostel available", "hostel chahiye", "on-campus living", "boarding"]

    for f in u.fields_offered:
        fl = f.lower()
        if "engineering" in fl:
            tags += ["engineering", "BE", "b.e.", "tech university"]
        if "computer" in fl:
            tags += ["CS", "computer science", "software", "IT", "programming"]
        if "medicine" in fl or "medical" in fl:
            tags += ["MBBS", "doctor", "medical college"]
        if "business" in fl:
            tags += ["BBA", "MBA", "commerce", "management"]
        if "arts" in fl:
            tags += ["arts", "humanities", "social sciences"]

    if u.hec_category in ("W1", "W2"):
        tags += ["top university", "ranked", "achhi university", "best uni", "prestigious"]
    if u.merit_cutoff and u.merit_cutoff <= 60:
        tags += ["low merit required", "easy admission", "kam marks", "60 percent"]

    city_prov_tags = {
        "punjab":    ["Punjab", "lahore multan faisalabad"],
        "sindh":     ["Sindh", "Karachi region", "hyderabad"],
        "islamabad": ["capital", "federal", "islamabad"],
    }
    for key, extra in city_prov_tags.items():
        if key in u.province.lower() or key in u.city.lower():
            tags += extra

    return "Keywords: " + ", ".join(tags) + "."


# ══════════════════════════════════════════════════════════════════════════════
# Metadata builder  (ChromaDB: str | int | float | bool only — no lists/None)
# ══════════════════════════════════════════════════════════════════════════════

def _build_metadata(u: UniversityRecord) -> dict:
    fields_pipe = "|".join(u.fields_offered).lower()
    tag_parts   = [u.city.lower(), u.province.lower(), u.type.lower(), u.hec_category.lower()]
    tag_parts  += [f.lower() for f in u.fields_offered]
    if u.scholarships: tag_parts.append("scholarship")
    if u.has_hostel:   tag_parts.append("hostel")
    if u.annual_fee_pkr and u.annual_fee_pkr <= 50_000: tag_parts.append("affordable")

    return {
        # Identity
        "university_id":    u.university_id,
        "name":             u.name,
        # Geography
        "city":             u.city,
        "province":         u.province,
        # Classification
        "type":             u.type,
        "hec_category":     u.hec_category,
        # Financials  (-1 = data not available)
        "annual_fee_pkr":   u.annual_fee_pkr  or -1,
        "has_scholarship":  bool(u.scholarships),
        # Academic
        "merit_cutoff":     u.merit_cutoff    or -1.0,
        "has_hostel":       u.has_hostel,
        "employment_rate":  u.employment_rate or -1,
        # Searchable strings
        "fields_pipe":      fields_pipe,
        "tags":             "|".join(tag_parts),
        # Full record blob — avoids a second DB round-trip on retrieval
        "record_json":      u.model_dump_json(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Collection management
# ══════════════════════════════════════════════════════════════════════════════

def get_collection(corpus: Optional[list[str]] = None) -> chromadb.Collection:
    """
    Get-or-create the HEC universities collection.
    corpus is used only when fitting the TF-IDF fallback.
    """
    if corpus is None:
        corpus = [_build_document(u) for u in _HEC_KB]
    backend = _get_embedding_backend(corpus=corpus)
    ef      = _TahqiqEmbeddingFunction(backend)
    client  = _get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={
            "hnsw:space":        "cosine",
            "description":       "HEC Pakistan university knowledge base — Tahqiq.ai",
            "embedding_backend": backend.name,
        },
    )


def add_universities(
    records: list[UniversityRecord],
    collection: Optional[chromadb.Collection] = None,
    skip_existing: bool = True,
) -> dict:
    """
    Embed and index UniversityRecord objects into ChromaDB.

    Returns:
        {"added": int, "skipped": int, "total": int, "backend": str}
    """
    col = collection or get_collection()

    existing_ids: set[str] = set()
    if skip_existing:
        try:
            existing_ids = set(col.get(include=[])["ids"])
        except Exception:
            pass

    to_add = [r for r in records if r.university_id not in existing_ids]
    if not to_add:
        logger.info("All %d records already indexed.", len(records))
        return {"added": 0, "skipped": len(records), "total": len(records),
                "backend": _embedding_backend.name if _embedding_backend else "unknown"}

    col.add(
        ids       =[r.university_id for r in to_add],
        documents =[_build_document(r) for r in to_add],
        metadatas =[_build_metadata(r) for r in to_add],
    )
    logger.info("✓ Indexed %d records: %s",
                len(to_add), [r.university_id for r in to_add])

    return {
        "added":   len(to_add),
        "skipped": len(existing_ids),
        "total":   len(records),
        "backend": _embedding_backend.name if _embedding_backend else "unknown",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Semantic search
# ══════════════════════════════════════════════════════════════════════════════

def _build_where_filter(intent: StudentIntent) -> Optional[dict]:
    """
    Build ChromaDB $where clause from intent.
    Only adds clauses for fields that are actually present — never guesses.
    """
    clauses: list[dict] = []

    if intent.max_fee_pkr:
        clauses.append({"$or": [
            {"annual_fee_pkr": {"$lte": intent.max_fee_pkr}},
            {"annual_fee_pkr": {"$eq": -1}},   # unknown fee — keep it
        ]})

    if intent.needs_hostel is True:
        clauses.append({"has_hostel": {"$eq": True}})

    if not clauses:    return None
    if len(clauses)==1: return clauses[0]
    return {"$and": clauses}


def semantic_search(
    query: str,
    intent: Optional[StudentIntent] = None,
    top_k: int = 3,
    collection: Optional[chromadb.Collection] = None,
    expand_if_filtered: bool = True,
) -> list[UniversityRecord]:
    """
    Hybrid semantic + metadata search against ChromaDB.

    Pipeline:
    1. Build $where filter from structured intent (fee ceiling, hostel).
    2. Vector similarity search (cosine) with optional filter.
    3. If filter yields < top_k results, expand with unfiltered search
       (filtered results keep priority).
    4. Post-rank boost for explicit field match.
    5. Deserialise metadata["record_json"] → typed UniversityRecord.

    Each record carries a `_search_score` float (0-1 cosine similarity).
    """
    col     = collection or get_collection()
    where   = _build_where_filter(intent) if intent else None
    fetch_n = max(top_k + 2, 5)

    def _run(where_clause: Optional[dict]) -> list[UniversityRecord]:
        kwargs: dict = {
            "query_texts": [query],
            "n_results":   fetch_n,
            "include":     ["metadatas", "distances", "documents"],
        }
        if where_clause:
            kwargs["where"] = where_clause
        try:
            res = col.query(**kwargs)
        except Exception as exc:
            logger.warning("ChromaDB query error: %s", exc)
            return []

        records: list[UniversityRecord] = []
        for meta, dist in zip(res.get("metadatas", [[]])[0],
                               res.get("distances",  [[]])[0]):
            try:
                rec = UniversityRecord(**json.loads(meta["record_json"]))
                rec._search_score = round(1.0 - dist, 4)
                records.append(rec)
            except Exception as exc:
                logger.warning("Deserialise error %s: %s",
                               meta.get("university_id"), exc)
        return records

    primary = _run(where)
    logger.info("semantic_search('%s') → %d hits [filter=%s, backend=%s]",
                query, len(primary),
                "yes" if where else "no",
                _embedding_backend.name if _embedding_backend else "?")

    # Expand if filtered results are insufficient
    if expand_if_filtered and where and len(primary) < top_k:
        logger.info("Expanding: %d/%d with filter → running unfiltered.", len(primary), top_k)
        seen = {r.university_id for r in primary}
        for r in _run(None):
            if r.university_id not in seen:
                primary.append(r)
                seen.add(r.university_id)
            if len(primary) >= top_k:
                break

    # Post-rank: promote explicit field matches
    if intent and intent.field:
        fl = intent.field.lower()
        primary.sort(key=lambda r: (
            0 if any(fl in f.lower() for f in r.fields_offered) else 1,
            -(getattr(r, "_search_score", 0)),
        ))

    return primary[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_collection_stats() -> dict:
    try:
        col   = get_collection()
        count = col.count()
        peek  = col.peek(limit=3)
        return {
            "status":            "healthy",
            "collection":        COLLECTION_NAME,
            "document_count":    count,
            "chroma_path":       str(CHROMA_PATH.resolve()),
            "embedding_backend": _embedding_backend.name if _embedding_backend else "not loaded",
            "embedding_model":   EMBEDDING_MODEL_NAME,
            "sample_ids":        peek.get("ids", []),
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def reset_collection() -> dict:
    """Delete and re-seed the collection. Dev use only."""
    global _embedding_backend
    _embedding_backend = None
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        cache = CHROMA_PATH / _TFIDFBackend._CACHE_FILE
        if cache.exists():
            cache.unlink()
        logger.warning("Collection '%s' deleted and TF-IDF cache cleared.", COLLECTION_NAME)
    except Exception:
        pass
    col    = get_collection()
    result = add_universities(_HEC_KB, collection=col, skip_existing=False)
    return {"reset": True, **result}


def seed_if_empty() -> dict:
    """
    Idempotent startup seeder — call on every FastAPI boot.
    Seeds _HEC_KB only when the collection is empty.
    """
    col   = get_collection()
    count = col.count()
    if count == 0:
        logger.info("Collection empty — seeding %d HEC records.", len(_HEC_KB))
        return add_universities(_HEC_KB, collection=col, skip_existing=False)
    logger.info("Collection has %d documents — seed skipped.", count)
    return {"added": 0, "skipped": count, "total": count}


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("\n🗄  Tahqiq.ai — ChromaDB Knowledge Base\n" + "─" * 55)

    print("\n[1] Seeding collection …")
    print("   ", seed_if_empty())

    print("\n[2] Collection stats …")
    for k, v in get_collection_stats().items():
        print(f"    {k}: {v}")

    test_queries = [
        "Sasti engineering uni in Multan?",
        "CS degree Lahore with scholarship",
        "Medical university Sindh low merit hostel chahiye",
        "Top ranked W1 university Pakistan",
    ]
    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]

    print("\n[3] Semantic search demos …\n")
    col = get_collection()
    for q in test_queries:
        print(f"  Query : '{q}'")
        results = semantic_search(q, top_k=3, collection=col)
        for i, r in enumerate(results, 1):
            score = getattr(r, "_search_score", 0)
            fee   = f"PKR {r.annual_fee_pkr:,}" if r.annual_fee_pkr else "N/A"
            print(f"    #{i} [{score:.3f}] {r.name[:45]:<45}"
                  f"  {r.city:<12} {r.hec_category}  fee={fee}")
        print()
