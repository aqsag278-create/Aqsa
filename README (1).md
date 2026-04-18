---
title: Tahqiq AI
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
license: mit
short_description: Pakistan's first Explainable University Advisor — Har Student Ka Apna University Guide
---

# Tahqiq.ai — Har Student Ka Apna University Guide

> **Pakistan's first Explainable University Intelligence System**
> Every Pakistani student's AI university advisor — free, forever.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange.svg)](https://langchain.com)
[![HEC Gen AI Hackathon 2025](https://img.shields.io/badge/HEC%20Gen%20AI-Hackathon%202025-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

500,000 Pakistani students make the most important decision of their lives every year — **most of them are guessing.**

Meet Bilal. He got 78% in FSc and wants to study Computer Science. His family saved Rs. 80,000 over three years — skipping weddings, selling gold — for one semester. He has one question that his family's entire future depends on:

> *"Baba ki mehnat zaaya na ho — mujhe konsi university mein jana chahiye?"*

The data to answer his question exists. HEC publishes rankings, scholarships, faculty strength, and graduate employment for every university. But it lives in fragmented English PDFs no rural student can navigate.

**Tahqiq.ai gives Bilal the answer his family deserves — in Urdu, in 30 seconds, free.**

---

## What It Does

Bilal types one Urdish query. Five AI agents respond in 30 seconds:

```
Input:  "Mere 78% hain, CS mein admission lena hai, scholarship chahiye,
         preferably Multan ke paas — koi acha university batao"

Output: #1 Bahauddin Zakariya University (BZU)
        City: Multan | Type: Public | HEC: W3
        Scholarship: HEC Ehsaas (100% tuition) — income < 500K PKR
        XAI: "BZU Multan mein hai aur aapke 78% ke saath admission
               ki achhi chances hain. ✅ City match, ✅ Public (affordable),
               ✅ Scholarship available. | Confidence: 92% (High)"
        Metrics: Affordability 88 | Merit Probability 75 | Market Value 62
        
        #2 ... #3 ...  (all with same depth)
        
        One-click PDF → share with parents
```

---

## Architecture — 5 Agents

```
Student Query (Urdish)
        │
        ▼
┌─────────────────────┐
│  Agent 1: Query     │  Intent extraction — Urdish/Roman Urdu/English
│  (Grok / LLM)       │  → StudentIntent: location, field, marks, budget
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 2: Data      │  CSV retrieval + deterministic scoring
│  (Pandas + HEC KB)  │  → Top 3 UniversityRecord from 209 universities
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 3: XAI       │  Urdish explanation — empathetic, data-cited
│  (Grok / LLM)       │  → xai_explanation + confidence score
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 4: Insights  │  Roman Urdu next-steps
│  (Grok / LLM)       │  → 3 actionable steps per university
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 5: Report    │  Strict ContractResponse + branded PDF
│  (Serialiser)       │  → One-click download for parents
└─────────────────────┘
```

**Plus: Multimodal OCR** — student uploads result card image → `grok-2-vision` extracts marks automatically.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Python FastAPI | Fast, lightweight, auto Swagger docs |
| Primary LLM | Grok (xAI) via LangChain | Best Urdish understanding, OpenAI-compatible |
| LLM Fallbacks | GPT-4o → Gemini → Claude | Never crashes — always responds |
| Vision / OCR | grok-2-vision-1212 | Result card mark extraction |
| Vector DB | ChromaDB + SentenceTransformers | Semantic search over 209 universities |
| Data Layer | Pandas + CSV (tahqiq_final_database.csv) | 209 HEC universities, all of Pakistan |
| PDF Reports | ReportLab | Branded one-click parent-ready report |
| Deployment | Docker → Hugging Face Spaces | Free hosting, zero config |

---

## Repository Structure

```
tahqiq_ai/
│
├── main.py                      # FastAPI app — all endpoints
├── agent_logic.py               # 5-agent pipeline + OCR + HEC threshold
├── llm_router.py                # Grok-first multi-provider LLM router
├── knowledge_base.py            # ChromaDB vector store (SentenceTransformer + TF-IDF)
│
├── tahqiq_final_database.csv    # 209 HEC universities — the knowledge base
├── contract.json                # Strict API response schema (reference)
│
├── Dockerfile                   # Hugging Face Spaces deployment
├── requirements.txt             # All Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Python / ML / secrets gitignore
└── README.md                    # This file
```

---

## Quick Start — Local Development

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/tahqiq-ai.git
cd tahqiq-ai
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env — add at least one key:
# XAI_API_KEY=your_xai_key        ← recommended (Grok)
# OPENAI_API_KEY=your_openai_key  ← fallback
```

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** — full interactive Swagger UI.

### 4. Test Bilal's query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Mere 78% hain, CS mein admission lena hai, scholarship chahiye, Multan ke paas",
    "percentage": 78,
    "city_pref": "Multan"
  }'
```

### 5. Test multimodal (result card image)

```bash
curl -X POST http://localhost:8000/query/multimodal \
  -F "query=Kaunsi university best hai?" \
  -F "city_pref=Lahore" \
  -F "image=@result_card.jpg"
```

---

## Hugging Face Spaces Deployment

### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name: `tahqiq-ai`
3. SDK: **Docker**
4. Visibility: Public

### Step 2 — Push your repo

```bash
# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/tahqiq-ai

# Push
git push hf main
```

### Step 3 — Set secrets

In your Space → **Settings → Repository secrets**, add:

| Secret | Value | Required |
|---|---|---|
| `XAI_API_KEY` | Your xAI key | ✅ Primary |
| `OPENAI_API_KEY` | Your OpenAI key | Optional fallback |
| `GEMINI_API_KEY` | Your Gemini key | Optional fallback |
| `ANTHROPIC_API_KEY` | Your Anthropic key | Optional fallback |

**That's it.** HF builds the Docker image automatically. Your Space URL:
`https://YOUR_USERNAME-tahqiq-ai.hf.space`

### Endpoints available on your Space

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Healthcheck |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/query` | JSON query → recommendations |
| `POST` | `/query/multimodal` | Image + query → OCR + recommendations |
| `GET` | `/download-report/{session_id}` | Branded PDF download |
| `GET` | `/kb/stats` | ChromaDB knowledge base stats |
| `GET` | `/llm/status` | LLM provider health |

---

## API Reference

### `POST /query`

```json
{
  "query":      "Mere 78% hain, CS mein admission lena hai",
  "percentage": 78.0,
  "field":      "Computer Science",
  "city_pref":  "Multan",
  "budget_pkr": 80000,
  "session_id": null
}
```

### `POST /query/multimodal` (multipart/form-data)

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Student's Urdish query |
| `image` | file | No | Result card / marksheet image |
| `percentage` | float | No | Override marks (ignored if image provided) |
| `city_pref` | string | No | City preference |
| `budget_pkr` | int | No | Max annual budget in PKR |

### Response shape

See `contract.json` for the complete annotated schema.

---

## HEC Threshold Validation

If a student's marks are below **45%** (HEC minimum for undergraduate admission), the API:

1. Still returns the best-match universities (for awareness)
2. Adds an `hec_eligibility` block with an honest Urdish message
3. Suggests 6 alternative pathways: ADP, AIOU, VU, TVET, etc.

```json
"hec_eligibility": {
  "eligible": false,
  "message": "Aapke 40% marks HEC ke minimum 45% se 5% kam hain...",
  "alternative_pathways": [
    "Associate Degree Program (ADP) 2 saal — phir degree mein transfer",
    "Allama Iqbal Open University (AIOU) — distance learning",
    "Virtual University of Pakistan — online programs",
    ...
  ]
}
```

---

## Knowledge Base

`tahqiq_final_database.csv` contains **209 unique universities** across all provinces:

| Province | Universities |
|---|---|
| Punjab | 76 rows (38 unique) |
| Khyber Pakhtunkhwa | 66 rows (33 unique) |
| Sindh | 46 rows (23 unique) |
| Islamabad Capital Territory | 39 rows (20 unique) |
| Balochistan | 19 rows (10 unique) |
| Azad Jammu & Kashmir | 6 rows (3 unique) |

**To re-enrich** the database with fresh HEC category / market value / affordability data:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run_enrichment.py          # full run (~4 min, ~$0.03)
python run_enrichment.py --limit 5  # test on 5 universities first
```

The script is crash-safe — re-run from where it stopped if interrupted.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `XAI_API_KEY` | — | xAI Grok API key (primary LLM) |
| `OPENAI_API_KEY` | — | OpenAI fallback |
| `GEMINI_API_KEY` | — | Gemini fallback |
| `ANTHROPIC_API_KEY` | — | Claude fallback |
| `LLM_PROVIDER` | `auto` | `auto` \| `grok` \| `openai` \| `gemini` \| `anthropic` |
| `LLM_TIMEOUT` | `30` | Per-call timeout seconds |
| `LLM_MAX_RETRIES` | `2` | Retries per provider |
| `HEC_CSV_PATH` | `./tahqiq_final_database.csv` | Path to university database |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory |

---

## Data Sources

All data is 100% public and verifiable:

| Source | Data |
|---|---|
| HEC Official Website | University rankings, category ratings |
| HEC Annual Reports | Scholarships, PhD output, faculty counts |
| Pakistan Bureau of Statistics | Enrollment, provincial distribution |

---

## License

MIT License — free to use, modify, and deploy. See [LICENSE](LICENSE).

---

## HEC Gen AI Hackathon 2026

*"Data ke paas jawab hain. Tahqiq.ai poochna jaanta hai."*
The data has answers. Tahqiq.ai knows how to ask.

**Tahqiq.ai · The Explainable Intelligence System · HEC Gen AI Hackathon 2025**
