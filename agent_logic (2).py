"""
Tahqiq.ai — agent_logic.py
Multi-Agent Pakistani Education Counselor

Pipeline:
  [Optional] Image → OCR Agent → extracted marks
       │
       ▼
  [Agent 1] Query Agent     →  StudentIntent  (Urdish intent extraction)
       │
       ▼
  [Agent 2] Data Agent      →  List[UniversityRecord]  (CSV/ChromaDB retrieval)
       │
       ▼
  [Agent 3] XAI Agent       →  Urdish explanation + confidence score
       │
       ▼
  [Agent 4] Insight Agent   →  next_steps (Roman Urdu action items)
       │
       ▼
  [Agent 5] Serialiser      →  ContractResponse  (strict contract shape)

Key features
────────────
• Grok-first via llm_router (XAI_API_KEY → OPENAI_API_KEY → GEMINI_API_KEY → ANTHROPIC_API_KEY)
• HEC 45% threshold check — flags below-threshold students, suggests pathways
• CSV-backed knowledge base (tahqiq_final_database.csv, 253 universities)
• Anti-hallucination: null for any field not confidently extracted
• Urdish XAI explanations: empathetic, data-cited, financially aware
• Multimodal: call extract_marks_from_image() for result-card OCR
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ValidationError

# ── shared Pydantic models from main ──────────────────────────────────────────
from main import (
    ContractResponse,
    Links,
    Metrics,
    RecommendationsData,
    Scholarship,
    UniversityRecommendation,
)

logger = logging.getLogger("tahqiq.agent")
logging.basicConfig(level=logging.INFO)

# ── HEC eligibility threshold ─────────────────────────────────────────────────
HEC_MIN_PERCENT = 45.0   # HEC minimum for undergraduate admission


# ══════════════════════════════════════════════════════════════════════════════
# LLM singleton — patched at startup by llm_router.patch_agent_logic()
# ══════════════════════════════════════════════════════════════════════════════

def _get_llm(temperature: float = 0.2):
    """
    Default LLM getter. This function is monkey-patched at startup by
    llm_router.patch_agent_logic() to use the full routing chain.
    Kept here as fallback for offline / test environments.
    """
    from llm_router import get_llm
    return get_llm(temperature)


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

class StudentIntent(BaseModel):
    location:             Optional[str]   = Field(None, description="City or region")
    field:                Optional[str]   = Field(None, description="Degree / discipline in English")
    max_fee_pkr:          Optional[int]   = Field(None, description="Max annual fee in PKR")
    merit_percent:        Optional[float] = Field(None, description="Student's FSc/Matric % (0-100)")
    needs_hostel:         Optional[bool]  = Field(None)
    scholarship_required: Optional[bool]  = Field(None)
    shift_pref:           Optional[str]   = Field(None, description="Morning / Evening / null")
    delivery_pref:        Optional[str]   = Field(None, description="Online / On-campus / null")
    uni_type_pref:        Optional[str]   = Field(None, description="Public / Private / null")
    raw_query:            str             = Field(...)
    confidence:           float           = Field(0.0)

    def missing_fields(self) -> list[str]:
        return [f for f in ["location", "field", "merit_percent"]
                if getattr(self, f) is None]

    def is_below_hec_threshold(self) -> bool:
        return (self.merit_percent is not None
                and self.merit_percent < HEC_MIN_PERCENT)


class UniversityRecord(BaseModel):
    """
    Unified university record — sourced from tahqiq_final_database.csv.
    """
    university_id:    str
    name:             str
    city:             str
    province:         str
    type:             str               # Public | Private | Private Non-Profit
    hec_category:     str               # W1 … W4 | X
    fields_offered:   list[str]         = Field(default_factory=list)
    annual_fee_pkr:   Optional[int]     = None
    merit_cutoff:     Optional[float]   = None
    has_hostel:       bool              = False
    employment_rate:  Optional[int]     = None
    scholarships:     list[dict]        = Field(default_factory=list)
    website:          str               = ""
    apply_url:        str               = ""
    shift:            Optional[str]     = None   # Morning | Evening
    delivery_mode:    Optional[str]     = None   # On-campus | Online | Blended
    market_value:     Optional[int]     = None   # 1-100 from enrichment
    affordability_rank: Optional[str]  = None   # Budget | Standard | Premium
    _search_score:    float             = 0.0    # populated by semantic_search


# ══════════════════════════════════════════════════════════════════════════════
# CSV → UniversityRecord loader  (seeds _HEC_KB at import time)
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv_kb(csv_path: Optional[str] = None) -> list[UniversityRecord]:
    """
    Load tahqiq_final_database.csv into a list of UniversityRecord.
    Handles:
    - MOR/EVE deduplication (keeps first occurrence per base university)
    - JSON-encoded scholarships column
    - Graceful handling of null fee / merit / metrics columns
    """
    import pandas as pd

    # Search order: explicit path → env var → local file → script directory
    candidates = [
        csv_path,
        os.environ.get("HEC_CSV_PATH"),
        "tahqiq_final_database.csv",
        str(Path(__file__).parent / "tahqiq_final_database.csv"),
    ]
    path = next((p for p in candidates if p and Path(p).exists()), None)

    if not path:
        logger.warning("tahqiq_final_database.csv not found — KB will be empty.")
        return []

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    # Extract base university ID (strip _MOR / _EVE suffix)
    df["_base"] = df["university_id"].str.extract(r"(U\d+[A-Za-z]*)")[0]
    df = df.drop_duplicates(subset="_base").reset_index(drop=True)
    logger.info("Deduplicated to %d unique universities", len(df))

    records: list[UniversityRecord] = []
    for _, row in df.iterrows():
        # Parse scholarships JSON
        try:
            scholarships = json.loads(row["scholarships_offered"]) \
                if pd.notna(row.get("scholarships_offered")) else []
        except (json.JSONDecodeError, TypeError):
            scholarships = []

        # Parse metrics JSON for any pre-computed values
        metrics_dict: dict = {}
        try:
            if pd.notna(row.get("metrics")):
                raw = str(row["metrics"]).replace("'", '"')
                metrics_dict = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Derive field list from university name heuristics
        fields = _infer_fields(str(row.get("name", "")), str(row.get("type", "")))

        # Fee: try fee_per_semester * 2, else None
        fee = None
        try:
            if pd.notna(row.get("fee_per_semester")):
                fee = int(float(row["fee_per_semester"]) * 2)
        except (TypeError, ValueError):
            pass

        # Market value from enrichment — guard against NaN float
        mkt = None
        try:
            mv_raw = row.get("market_value")
            if mv_raw is not None and pd.notna(mv_raw):
                mkt = int(float(mv_raw))
        except (TypeError, ValueError):
            pass

        # hec_category — default W3 for unenriched rows (NaN)
        try:
            hec_raw = row.get("hec_category", "W3")
            hec_cat = "W3" if (hec_raw is None or pd.isna(hec_raw)) else str(hec_raw)
        except (TypeError, ValueError):
            hec_cat = "W3"

        # affordability_rank — guard against NaN
        try:
            af_raw = row.get("affordability_rank", "")
            afford_str = "" if (af_raw is None or pd.isna(af_raw)) else str(af_raw)
        except (TypeError, ValueError):
            afford_str = ""

        rec = UniversityRecord(
            university_id    = str(row.get("_base", row.get("university_id", ""))),
            name             = str(row.get("name", "Unknown")),
            city             = str(row.get("city", "")),
            province         = str(row.get("province", "")),
            type             = str(row.get("type", "Public")),
            hec_category     = hec_cat,
            fields_offered   = fields,
            annual_fee_pkr   = fee,
            merit_cutoff     = None,     # not in current CSV
            has_hostel       = True,     # conservative default
            employment_rate  = None,
            scholarships     = scholarships,
            website          = str(row.get("website", row.get("prospectus_url", ""))),
            apply_url        = str(row.get("prospectus_url", "")),
            shift            = str(row.get("shift", "")) or None,
            delivery_mode    = str(row.get("delivery_mode", "")) or None,
            market_value     = mkt,
            affordability_rank = afford_str or None,
        )
        records.append(rec)

    logger.info("KB loaded: %d university records", len(records))
    return records


def _infer_fields(name: str, uni_type: str) -> list[str]:
    """Heuristic field inference from university name — avoids empty lists."""
    name_l  = name.lower()
    fields: list[str] = []

    if any(k in name_l for k in ["engineering", "technology", "tech"]):
        fields += ["Engineering", "Computer Science"]
    if any(k in name_l for k in ["computer", "computing", "nuces", "fast", "vu", "virtual"]):
        fields.append("Computer Science")
    if any(k in name_l for k in ["medical", "medicine", "health", "shifa", "dow", "aga khan"]):
        fields += ["Medicine", "Health Sciences"]
    if any(k in name_l for k in ["management", "commerce", "business", "iba", "lums", "iobm"]):
        fields += ["Business", "Commerce", "MBA"]
    if any(k in name_l for k in ["law", "legal"]):
        fields.append("Law")
    if any(k in name_l for k in ["agriculture", "agri"]):
        fields.append("Agriculture")
    if any(k in name_l for k in ["art", "design", "architecture"]):
        fields.append("Arts & Design")
    if any(k in name_l for k in ["education", "teacher"]):
        fields.append("Education")
    if any(k in name_l for k in ["pharmacy", "pharma"]):
        fields.append("Pharmacy")
    if any(k in name_l for k in ["islamic", "urdu", "language", "modern language"]):
        fields += ["Arts", "Humanities", "Languages"]
    if any(k in name_l for k in ["open university", "allama iqbal", "aiou"]):
        fields += ["Distance Education", "Computer Science", "Arts"]

    # If nothing matched, general university
    if not fields:
        fields = ["General Studies", "Computer Science", "Engineering", "Arts"]

    return list(dict.fromkeys(fields))  # deduplicate preserving order


# Seed on module import — lazy, crash-safe
_HEC_KB: list[UniversityRecord] = _load_csv_kb()


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — Query Agent: Urdish intent extraction
# ══════════════════════════════════════════════════════════════════════════════

_EXTRACTOR_SYSTEM = """
You are a bilingual (Urdu/English) intent-extraction engine for Tahqiq.ai,
Pakistan's AI university advisor.

Extract structured information from student queries written in Roman Urdu,
formal Urdu, English, or any mix ("Urdish").

Output ONLY a valid JSON object — no markdown, no explanation, no preamble.

JSON schema (null for any field you cannot confidently extract — DO NOT guess):
{
  "location":             string | null,
  "field":                string | null,
  "max_fee_pkr":          integer | null,
  "merit_percent":        float | null,
  "needs_hostel":         boolean | null,
  "scholarship_required": boolean | null,
  "shift_pref":           "Morning" | "Evening" | null,
  "delivery_pref":        "Online" | "On-campus" | null,
  "uni_type_pref":        "Public" | "Private" | null,
  "confidence":           float
}

Urdish vocabulary:
- "sasti / sasta / kam fees" → affordability signal
- "scholarship chahiye"      → scholarship_required = true
- "hostel chahiye / bahar se hun" → needs_hostel = true
- "subah / morning"         → shift_pref = "Morning"
- "shaam / evening"         → shift_pref = "Evening"
- "online / ghar se"        → delivery_pref = "Online"
- "sarkari"                 → uni_type_pref = "Public"
- Percentage/marks          → merit_percent

Return ONLY the JSON object.
"""


def extract_intent(query: str, llm=None) -> StudentIntent:
    """Agent 1: Parse Urdish query → StudentIntent."""
    llm = llm or _get_llm(temperature=0.0)
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=_EXTRACTOR_SYSTEM),
            HumanMessage(content=f"Student query: {query}"),
        ])
        chain    = prompt | llm | StrOutputParser()
        raw_json = chain.invoke({})
        raw_json = re.sub(r"```(?:json)?|```", "", raw_json).strip()
        data     = json.loads(raw_json)
        data["raw_query"] = query
        intent = StudentIntent(**data)
        logger.info("Intent: %s", intent.model_dump())
        return intent
    except Exception as exc:
        logger.warning("Intent extraction failed (%s) — rule-based fallback", exc)
        return _rule_based_extract(query)


def _rule_based_extract(query: str) -> StudentIntent:
    """Fast regex/keyword extractor — no LLM, always available."""
    q = query.lower()

    location = None
    for city in ["multan", "lahore", "islamabad", "karachi", "peshawar",
                 "quetta", "faisalabad", "hyderabad", "rawalpindi", "gujranwala",
                 "sialkot", "abbottabad", "bahawalpur", "sargodha", "di khan"]:
        if city in q:
            location = city.title()
            break

    # Province fallback
    if not location:
        for prov, pname in [("punjab", "Punjab"), ("sindh", "Sindh"),
                             ("kpk", "Peshawar"), ("balochistan", "Quetta")]:
            if prov in q:
                location = pname
                break

    field = None
    for f, kws in [
        ("Computer Science",  ["cs", "computer", "software", "it ", "programming"]),
        ("Engineering",       ["engineering", "be ", "beng", "b.e", "electrical", "mechanical", "civil"]),
        ("Medicine",          ["mbbs", "medical", "medicine", "doctor", "bds"]),
        ("Business",          ["bba", "mba", "commerce", "business", "accounting", "management"]),
        ("Arts",              ["arts", "humanities", "social"]),
        ("Law",               ["law", "llb"]),
        ("Pharmacy",          ["pharmacy", "pharma", "pharm"]),
    ]:
        if any(k in q for k in kws):
            field = f
            break

    merit = None
    m = re.search(r"(\d{2,3})\s*(?:%|percent|marks|score)", query, re.IGNORECASE)
    if m:
        merit = float(m.group(1))

    max_fee = None
    fm = re.search(r"(\d[\d,]+)\s*(?:pkr|rs\.?|rupees?)", q)
    if fm:
        try:
            max_fee = int(fm.group(1).replace(",", ""))
        except ValueError:
            pass

    scholarship  = any(w in q for w in ["scholarship", "scholars", "wazifa", "grant", "free"])
    hostel       = any(w in q for w in ["hostel", "accommodation", "stay", "bahar se"])
    sasti        = any(w in q for w in ["sasti", "sasta", "cheap", "affordable", "kam fee", "low fee"])
    if sasti and max_fee is None:
        max_fee  = 50_000

    shift = None
    if any(w in q for w in ["morning", "subah"]):
        shift = "Morning"
    elif any(w in q for w in ["evening", "shaam"]):
        shift = "Evening"

    delivery = None
    if any(w in q for w in ["online", "ghar se", "distance"]):
        delivery = "Online"

    uni_type = None
    if any(w in q for w in ["sarkari", "government", "public"]):
        uni_type = "Public"
    elif any(w in q for w in ["private"]):
        uni_type = "Private"

    return StudentIntent(
        location=location, field=field, max_fee_pkr=max_fee,
        merit_percent=merit, needs_hostel=hostel or None,
        scholarship_required=scholarship or None,
        shift_pref=shift, delivery_pref=delivery, uni_type_pref=uni_type,
        raw_query=query, confidence=0.6,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Data Agent: retrieve & rank universities
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_universities(intent: StudentIntent, top_k: int = 3) -> list[UniversityRecord]:
    """
    Agent 2: Score and rank universities from _HEC_KB.
    Deterministic scoring — no LLM — pure data-driven filtering.
    """
    pool = list(_HEC_KB)

    # Hard filters (relaxed on empty result)
    def _apply_filters(records: list[UniversityRecord]) -> list[UniversityRecord]:
        res = records

        if intent.location:
            loc = intent.location.lower()
            filtered = [r for r in res
                        if loc in r.city.lower() or loc in r.province.lower()]
            if filtered:
                res = filtered

        if intent.uni_type_pref:
            filtered = [r for r in res
                        if intent.uni_type_pref.lower() in r.type.lower()]
            if filtered:
                res = filtered

        if intent.shift_pref:
            filtered = [r for r in res
                        if r.shift and intent.shift_pref.lower() == r.shift.lower()]
            if filtered:
                res = filtered

        if intent.delivery_pref:
            filtered = [r for r in res
                        if r.delivery_mode and intent.delivery_pref.lower() in r.delivery_mode.lower()]
            if filtered:
                res = filtered

        if intent.max_fee_pkr:
            filtered = [r for r in res
                        if r.annual_fee_pkr is None or r.annual_fee_pkr <= intent.max_fee_pkr]
            if filtered:
                res = filtered

        if intent.needs_hostel:
            filtered = [r for r in res if r.has_hostel]
            if filtered:
                res = filtered

        return res

    candidates = _apply_filters(pool)
    if not candidates:
        candidates = pool  # full fallback — never return empty

    # Scoring
    def _score(r: UniversityRecord) -> float:
        s = 0.0

        # Field match
        if intent.field:
            fl = intent.field.lower()
            if any(fl in f.lower() for f in r.fields_offered):
                s += 30.0

        # HEC category prestige
        cat_score = {"W1": 25, "W2": 20, "W3": 12, "W4": 6, "X": 2}
        s += cat_score.get(r.hec_category.upper(), 5)

        # Market value (if available from enrichment)
        if r.market_value:
            s += r.market_value * 0.15

        # Scholarship match
        if intent.scholarship_required and r.scholarships:
            s += 15.0

        # Fee affordability
        if intent.max_fee_pkr and r.annual_fee_pkr:
            if r.annual_fee_pkr <= intent.max_fee_pkr * 0.75:
                s += 10.0
            elif r.annual_fee_pkr <= intent.max_fee_pkr:
                s += 5.0

        # Public uni preference for budget students
        if intent.max_fee_pkr and intent.max_fee_pkr <= 60_000 and "public" in r.type.lower():
            s += 8.0

        return s

    candidates.sort(key=_score, reverse=True)
    return candidates[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — XAI Agent: Urdish explanation + confidence
# ══════════════════════════════════════════════════════════════════════════════

_XAI_SYSTEM = """
Tu Tahqiq.ai ka empathetic Pakistani university counselor hai.
Ek student ke liye ek university ki recommendation explain kar.

Rules (strict):
1. Roman Urdu (Urdish) mein likho — English mix allowed.
2. Specific data cite karo: HEC category, fees, scholarships, employment.
3. Agar student ke marks kam hain, honest raho lekin encouraging bhi.
4. Agar fees zyada hai budget se, seedha batao — alternatives suggest karo.
5. 3-5 sentences maximum. ✅ for strengths, ⚠️ for honest caveats.
6. "Yeh info available nahi" bol do agar data nahi — KABHI hallucinate mat karo.
7. Bhai/behan jaise baat karo — not formal corporate language.
"""

_XAI_HUMAN = """
Student profile:
- Marks: {merit}
- Field: {field}
- City preference: {city}
- Scholarship needed: {schol}
- Max budget: {budget}

University:
- Name: {uni_name}
- City: {uni_city}
- Type: {uni_type}
- HEC Category: {hec_cat}
- Annual Fee: {fee}
- Scholarships: {scholarships}
- Employment rate: {employment}
- Fields offered: {fields}

Explain in Urdish WHY this university suits this student. Be empathetic and specific.
"""


def generate_xai_explanation(intent: StudentIntent, uni: UniversityRecord,
                              llm=None) -> str:
    """Agent 3: Generate empathetic Urdish XAI explanation via LLM."""
    llm = llm or _get_llm(temperature=0.4)

    fee_str = f"PKR {uni.annual_fee_pkr:,}/year" if uni.annual_fee_pkr else "information available nahi"
    sc_str  = "; ".join(s.get("name", "") for s in uni.scholarships[:2]) or "listed nahi"
    emp_str = f"{uni.employment_rate}% employment" if uni.employment_rate else "data available nahi"

    try:
        prompt  = ChatPromptTemplate.from_messages([
            SystemMessage(content=_XAI_SYSTEM),
            HumanMessage(content=_XAI_HUMAN.format(
                merit      = f"{intent.merit_percent}%" if intent.merit_percent else "not mentioned",
                field      = intent.field or "not specified",
                city       = intent.location or "any",
                schol      = "Yes" if intent.scholarship_required else "No",
                budget     = f"PKR {intent.max_fee_pkr:,}" if intent.max_fee_pkr else "not specified",
                uni_name   = uni.name,
                uni_city   = uni.city,
                uni_type   = uni.type,
                hec_cat    = uni.hec_category,
                fee        = fee_str,
                scholarships = sc_str,
                employment = emp_str,
                fields     = ", ".join(uni.fields_offered[:4]),
            )),
        ])
        chain  = prompt | llm | StrOutputParser()
        result = chain.invoke({})
        return result.strip()
    except Exception as exc:
        logger.warning("XAI generation failed for '%s': %s — using fallback", uni.name, exc)
        return _fallback_explanation(intent, uni)


def _fallback_explanation(intent: StudentIntent, uni: UniversityRecord) -> str:
    """Deterministic fallback when LLM is unavailable — data-only, no hallucination."""
    fee_str = f"PKR {uni.annual_fee_pkr:,}/year" if uni.annual_fee_pkr else "fee info available nahi"
    emp_str = f"{uni.employment_rate}%" if uni.employment_rate else "available nahi"
    sc_name = uni.scholarships[0].get("name", "") if uni.scholarships else "koi scholarship listed nahi"

    merit_note = ""
    if intent.merit_percent and uni.merit_cutoff:
        gap = intent.merit_percent - uni.merit_cutoff
        if gap >= 0:
            merit_note = f"Aapke {intent.merit_percent}% ke saath admission possible lag raha hai. "
        else:
            merit_note = (
                f"Honest baat: last year ka cutoff {uni.merit_cutoff}% tha "
                f"aur aapke {intent.merit_percent}% se {abs(gap):.1f}% ka gap hai. "
            )

    return (
        f"{uni.name} {uni.city} mein hai — HEC category {uni.hec_category}. "
        f"Annual fee approximately {fee_str} hai. {merit_note}"
        f"Graduates ki employment rate {emp_str} hai. "
        f"Scholarship option: {sc_name}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Agent 4 — Insight Agent: next steps in Roman Urdu
# ══════════════════════════════════════════════════════════════════════════════

_STEPS_SYSTEM = """
Tu Pakistani student ka guide hai. 3 actionable next steps do Roman Urdu mein.
Return ONLY a JSON array: ["step1", "step2", "step3"]
Steps should be specific: apply link, deadline check, scholarship application etc.
"""


def generate_next_steps(uni: UniversityRecord, intent: StudentIntent,
                        llm=None) -> list[str]:
    """Agent 4: Generate Roman Urdu action steps for the student."""
    llm = llm or _get_llm(temperature=0.3)
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=_STEPS_SYSTEM),
            HumanMessage(content=(
                f"University: {uni.name}, URL: {uni.apply_url}, "
                f"Scholarship needed: {intent.scholarship_required}, "
                f"HEC category: {uni.hec_category}"
            )),
        ])
        chain  = prompt | llm | StrOutputParser()
        raw    = chain.invoke({})
        raw    = re.sub(r"```(?:json)?|```", "", raw).strip()
        steps  = json.loads(raw)
        if isinstance(steps, list):
            return [str(s) for s in steps[:3]]
    except Exception as exc:
        logger.warning("next_steps failed for '%s': %s", uni.name, exc)

    # Fallback
    return [
        f"{uni.name} ki official website visit karo: {uni.website or uni.apply_url}",
        "Admission form download karo aur deadlines check karo.",
        "HEC Ehsaas scholarship ke liye income certificate tayar karo.",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# XAI confidence scorer
# ══════════════════════════════════════════════════════════════════════════════

def _compute_confidence(intent: StudentIntent, uni: UniversityRecord) -> dict:
    """Deterministic confidence score — fully auditable arithmetic."""
    criteria_met = 0
    total        = 0

    # Field match
    total += 1
    if intent.field and any(intent.field.lower() in f.lower()
                            for f in uni.fields_offered):
        criteria_met += 1

    # Location
    total += 1
    if intent.location and (
        intent.location.lower() in uni.city.lower()
        or intent.location.lower() in uni.province.lower()
    ):
        criteria_met += 1

    # Scholarship
    if intent.scholarship_required:
        total += 1
        if uni.scholarships:
            criteria_met += 1

    # Fee
    if intent.max_fee_pkr:
        total += 1
        if uni.annual_fee_pkr is None or uni.annual_fee_pkr <= intent.max_fee_pkr:
            criteria_met += 1

    ratio     = (criteria_met / total) if total else 0.5
    cat_bonus = {"W1": 0.08, "W2": 0.05, "W3": 0.02, "W4": 0.0, "X": -0.05}
    score     = min(0.97, ratio + cat_bonus.get(uni.hec_category.upper(), 0.0))

    return {
        "overall_confidence": round(score, 2),
        "criteria_met":       criteria_met,
        "criteria_total":     total,
        "level":              "High" if score >= 0.75 else "Medium" if score >= 0.50 else "Low",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Agent 5 — Contract Serialiser
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(intent: StudentIntent, uni: UniversityRecord) -> Metrics:
    """Fully deterministic metrics — no LLM involvement."""
    # Affordability score
    if uni.annual_fee_pkr is None:
        afford = 50
    elif uni.annual_fee_pkr <= 30_000:
        afford = 92
    elif uni.annual_fee_pkr <= 60_000:
        afford = 75
    elif uni.annual_fee_pkr <= 120_000:
        afford = 55
    elif uni.annual_fee_pkr <= 200_000:
        afford = 35
    else:
        afford = 18

    if intent.max_fee_pkr and uni.annual_fee_pkr:
        if uni.annual_fee_pkr > intent.max_fee_pkr:
            afford = max(5, afford - 20)

    # Merit probability
    if uni.merit_cutoff is None:
        merit_prob = 55
    elif intent.merit_percent is None:
        merit_prob = 50
    else:
        gap = intent.merit_percent - uni.merit_cutoff
        if gap >= 10:
            merit_prob = 88
        elif gap >= 5:
            merit_prob = 75
        elif gap >= 0:
            merit_prob = 62
        elif gap >= -5:
            merit_prob = 35
        else:
            merit_prob = 12

    # Market value — use enriched value or category heuristic
    if uni.market_value:
        mkt = uni.market_value
    else:
        cat_mkt = {"W1": 92, "W2": 78, "W3": 58, "W4": 35, "X": 15}
        mkt = cat_mkt.get(uni.hec_category.upper(), 50)

    return Metrics(
        affordability_score = max(0, min(100, afford)),
        merit_probability   = max(0, min(100, merit_prob)),
        market_value        = max(0, min(100, mkt)),
    )


def serialize_to_contract(
    intent: StudentIntent,
    universities: list[UniversityRecord],
    explanations: list[str],
    next_steps_list: Optional[list[list[str]]] = None,
) -> ContractResponse:
    """Agent 5: Assemble strict ContractResponse from all agent outputs."""
    recs: list[UniversityRecommendation] = []

    for i, (uni, xai) in enumerate(zip(universities, explanations)):
        conf     = _compute_confidence(intent, uni)
        metrics  = _compute_metrics(intent, uni)
        steps    = (next_steps_list[i] if next_steps_list and i < len(next_steps_list)
                    else [])

        scholarships = [
            Scholarship(
                name     = s.get("name", "Scholarship"),
                criteria = s.get("criteria", "See university website"),
                coverage = s.get("coverage", "Partial"),
            )
            for s in uni.scholarships[:3]
        ] or [
            Scholarship(
                name     = "HEC Ehsaas Undergraduate Scholarship",
                criteria = "Income < PKR 500,000/year",
                coverage = "100% Tuition",
            )
        ]

        # Embed confidence + next_steps in xai_explanation if space permits
        conf_tag  = f" | Confidence: {int(conf['overall_confidence']*100)}% ({conf['level']})"
        steps_tag = (" | Next steps: " + "; ".join(steps[:2])) if steps else ""
        full_xai  = xai + conf_tag + steps_tag

        website  = uni.website or uni.apply_url or f"https://hec.gov.pk"
        apply_url = uni.apply_url or website

        recs.append(UniversityRecommendation(
            university_id        = uni.university_id,
            name                 = uni.name,
            city                 = uni.city,
            type                 = uni.type,
            hec_category         = uni.hec_category,
            xai_explanation      = full_xai,
            scholarships_offered = scholarships,
            metrics              = metrics,
            links                = Links(website=website, apply=apply_url),
        ))

    return ContractResponse(
        status = "success",
        data   = RecommendationsData(recommendations=recs),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Multimodal — OCR Agent: extract marks from result card image
# ══════════════════════════════════════════════════════════════════════════════

_OCR_PROMPT = """
This is a Pakistani student's result card or marksheet image.
Extract ONLY the following as a JSON object:
{
  "marks_percent": float | null,   // overall percentage if shown
  "total_marks":   int | null,     // total marks obtained
  "max_marks":     int | null,     // maximum possible marks
  "board":         string | null,  // examining board name
  "year":          int | null,     // year of result
  "grade":         string | null   // grade/division if shown
}
If a value is not clearly visible, use null. Return ONLY the JSON object.
"""


def extract_marks_from_image(image_b64: str, mime_type: str = "image/jpeg") -> dict:
    """
    OCR Agent: Extract student marks from a result card image.

    Uses Grok vision (grok-2-vision-1212) or best available vision model.
    Returns a dict with keys: marks_percent, total_marks, max_marks, board, year, grade.
    All values may be None if extraction fails.
    """
    from llm_router import call_vision

    logger.info("OCR: extracting marks from image (%s)", mime_type)
    raw = call_vision(image_b64, mime_type, _OCR_PROMPT)

    if not raw:
        logger.warning("OCR returned empty — returning defaults")
        return _empty_ocr_result()

    # Parse JSON from response
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    if not clean.startswith("{"):
        m = re.search(r"\{.*?\}", clean, re.DOTALL)
        clean = m.group(0) if m else "{}"

    try:
        result = json.loads(clean)
        # Compute percentage from totals if not directly given
        if result.get("marks_percent") is None:
            tm = result.get("total_marks")
            mm = result.get("max_marks")
            if tm and mm and mm > 0:
                result["marks_percent"] = round(tm / mm * 100, 2)
        logger.info("OCR result: %s", result)
        return result
    except json.JSONDecodeError:
        logger.warning("OCR JSON parse failed: %s", raw[:200])
        return _empty_ocr_result()


def _empty_ocr_result() -> dict:
    return {
        "marks_percent": None,
        "total_marks":   None,
        "max_marks":     None,
        "board":         None,
        "year":          None,
        "grade":         None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEC eligibility gate
# ══════════════════════════════════════════════════════════════════════════════

def check_hec_eligibility(merit_percent: float) -> dict:
    """
    Check if student meets HEC 45% minimum threshold.
    Returns eligibility status + guidance if below threshold.
    """
    if merit_percent >= HEC_MIN_PERCENT:
        return {"eligible": True, "message": None, "alternative_pathways": []}

    gap = HEC_MIN_PERCENT - merit_percent
    return {
        "eligible": False,
        "message": (
            f"Aapke {merit_percent}% marks HEC ke minimum {HEC_MIN_PERCENT}% se "
            f"{gap:.1f}% kam hain. Regular degree programs mein seedha admission "
            f"mushkil hoga. Lekin fikr mat karo — alternatives available hain."
        ),
        "alternative_pathways": [
            "Associate Degree Program (ADP) 2 saal — phir degree mein transfer",
            "Diploma in IT / Business Administration (no merit bar)",
            "Allama Iqbal Open University (AIOU) — distance learning, flexible entry",
            "Virtual University of Pakistan — online programs, accessible entry",
            "Technical & Vocational Education (TVET) — practical skills + income faster",
            "Matric ya FSc improvement exam do — marks improve karo aur aglay saal apply karo",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════════

class AgentResult(BaseModel):
    intent:              StudentIntent
    missing_fields:      list[str]
    data_warning:        Optional[str]
    hec_eligibility:     Optional[dict]
    contract:            ContractResponse


def run_agent(raw_query: str, merit_override: Optional[float] = None) -> AgentResult:
    """
    Full pipeline: Urdish query → AgentResult (contract + metadata).

    Args:
        raw_query:      Student's query in Urdish/English.
        merit_override: Pre-extracted marks (e.g., from OCR). Overrides query parsing.
    """
    llm = _get_llm()

    # Step 1 — Intent
    logger.info("── Agent 1: Intent extraction ─────────────────────")
    intent = extract_intent(raw_query, llm=llm)

    if merit_override is not None:
        intent.merit_percent = merit_override
        logger.info("Merit override from OCR: %.1f%%", merit_override)

    # HEC eligibility check
    hec_check = None
    if intent.merit_percent is not None:
        hec_check = check_hec_eligibility(intent.merit_percent)
        if not hec_check["eligible"]:
            logger.warning("Student below HEC threshold: %.1f%%", intent.merit_percent)

    missing      = intent.missing_fields()
    data_warning = None
    if missing:
        data_warning = (
            f"Query mein yeh info nahi mili: {', '.join(missing)}. "
            f"Broad criteria se recommendations ban rahi hain. "
            f"Zyada specific query se better results milenge."
        )

    # Step 2 — Data
    logger.info("── Agent 2: University retrieval ──────────────────")
    universities = retrieve_universities(intent, top_k=3)
    logger.info("Retrieved: %s", [u.university_id for u in universities])

    if not universities:
        return AgentResult(
            intent=intent, missing_fields=missing, data_warning=data_warning,
            hec_eligibility=hec_check,
            contract=ContractResponse(
                status="no_results",
                data=RecommendationsData(recommendations=[]),
            ),
        )

    # Step 3 — XAI explanations
    logger.info("── Agent 3: XAI explanations ──────────────────────")
    explanations = [generate_xai_explanation(intent, u, llm=llm)
                    for u in universities]

    # Step 4 — Next steps
    logger.info("── Agent 4: Next steps ────────────────────────────")
    next_steps = [generate_next_steps(u, intent, llm=llm) for u in universities]

    # Step 5 — Serialize
    logger.info("── Agent 5: Serialising contract ──────────────────")
    contract = serialize_to_contract(intent, universities, explanations, next_steps)

    return AgentResult(
        intent=intent,
        missing_fields=missing,
        data_warning=data_warning,
        hec_eligibility=hec_check,
        contract=contract,
    )


def run_agent_offline(raw_query: str) -> AgentResult:
    """Steps 2-5 only, no LLM — for CI/testing."""
    intent       = _rule_based_extract(raw_query)
    universities = retrieve_universities(intent, top_k=3)
    explanations = [_fallback_explanation(intent, u) for u in universities]
    contract     = serialize_to_contract(intent, universities, explanations)
    hec_check    = check_hec_eligibility(intent.merit_percent) \
                   if intent.merit_percent else None
    return AgentResult(
        intent=intent,
        missing_fields=intent.missing_fields(),
        data_warning=None,
        hec_eligibility=hec_check,
        contract=contract,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Mere 78% hain, CS mein admission lena hai, scholarship chahiye, Multan ke paas"

    print(f"\nTahqiq.ai Agent — Query: '{query}'\n{'─'*60}")
    use_llm = bool(os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                   or os.environ.get("ANTHROPIC_API_KEY"))

    result = run_agent(query) if use_llm else run_agent_offline(query)
    print(f"\nIntent:\n{result.intent.model_dump_json(indent=2)}")
    if result.data_warning:
        print(f"\nWarning: {result.data_warning}")
    if result.hec_eligibility and not result.hec_eligibility["eligible"]:
        print(f"\nHEC Check: {result.hec_eligibility['message']}")
    print(f"\nContract:\n{result.contract.model_dump_json(indent=2)}")
