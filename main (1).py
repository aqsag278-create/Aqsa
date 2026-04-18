"""
Tahqiq.ai — main.py
FastAPI Backend · HEC Gen AI Hackathon 2025
Har Student Ka Apna University Guide

Endpoints
─────────
  GET  /                          healthcheck
  GET  /kb/stats                  ChromaDB stats
  GET  /llm/status                LLM router status
  POST /llm/reset/{provider}      reset provider failure counter
  POST /kb/reset                  wipe + reseed ChromaDB
  POST /query                     JSON query → recommendations
  POST /query/multimodal          Multipart/Form-Data: image + query → OCR + recommendations
  GET  /download-report/{sid}     branded PDF report

Deployment: Hugging Face Spaces (see Dockerfile)
"""

import base64
import io
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic contract models  (contract.json shape — strict, never change)
# ══════════════════════════════════════════════════════════════════════════════

class Scholarship(BaseModel):
    name:     str
    criteria: str
    coverage: str


class Links(BaseModel):
    website: str
    apply:   str


class Metrics(BaseModel):
    affordability_score: int = Field(..., ge=0, le=100)
    merit_probability:   int = Field(..., ge=0, le=100)
    market_value:        int = Field(..., ge=0, le=100)


class UniversityRecommendation(BaseModel):
    university_id:        str
    name:                 str
    city:                 str
    type:                 str        # Public / Private
    hec_category:         str
    xai_explanation:      str
    scholarships_offered: list[Scholarship]
    metrics:              Metrics
    links:                Links


class RecommendationsData(BaseModel):
    recommendations: list[UniversityRecommendation]


class ContractResponse(BaseModel):
    status: str
    data:   RecommendationsData


# ── Request / Response models ──────────────────────────────────────────────

class StudentProfile(BaseModel):
    query:      str             = Field(..., example="Mere 78% hain, CS padhna chahta hun")
    percentage: Optional[float] = Field(None, ge=0, le=100)
    field:      Optional[str]   = Field(None, example="Computer Science")
    city_pref:  Optional[str]   = Field(None, example="Lahore")
    budget_pkr: Optional[int]   = Field(None, example=80_000)
    session_id: Optional[str]   = None


class QueryResponse(BaseModel):
    session_id:      str
    query_echo:      str
    timestamp:       str
    response:        ContractResponse
    data_warning:    Optional[str]   = None
    hec_eligibility: Optional[dict]  = None
    ocr_result:      Optional[dict]  = None


# ══════════════════════════════════════════════════════════════════════════════
# In-memory session store  (swap for Redis in production)
# ══════════════════════════════════════════════════════════════════════════════

session_store: dict[str, QueryResponse] = {}

# ══════════════════════════════════════════════════════════════════════════════
# Module availability flags
# ══════════════════════════════════════════════════════════════════════════════

_KB_AVAILABLE    = False
_AGENT_AVAILABLE = False


def _try_import_modules():
    """Lazy import so the app boots even if optional deps are missing."""
    global _KB_AVAILABLE, _AGENT_AVAILABLE
    try:
        # Agent logic
        import agent_logic  # noqa: F401
        _AGENT_AVAILABLE = True
    except Exception as e:
        print(f"[Tahqiq.ai] agent_logic unavailable: {e}")

    try:
        # Knowledge base
        import knowledge_base  # noqa: F401
        _KB_AVAILABLE = True
    except Exception as e:
        print(f"[Tahqiq.ai] knowledge_base unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PDF report generator
# ══════════════════════════════════════════════════════════════════════════════

def _generate_pdf(session: QueryResponse) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf  = io.BytesIO()
    NAVY = colors.HexColor("#0A1628")
    GOLD = colors.HexColor("#F5A623")
    TEAL = colors.HexColor("#00B4D8")
    LITE = colors.HexColor("#F8F9FA")
    GREY = colors.HexColor("#6C757D")

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=14*mm,  bottomMargin=16*mm)
    W   = A4[0] - 36*mm

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    story = []

    # Header banner
    hdr_data = [[
        Paragraph("Tahqiq.ai",
                  S("T", fontName="Helvetica-Bold", fontSize=22,
                    textColor=colors.white, alignment=TA_CENTER)),
        Paragraph("Har Student Ka Apna University Guide",
                  S("ST", fontName="Helvetica", fontSize=10,
                    textColor=GOLD, alignment=TA_CENTER)),
    ]]
    hdr = Table(hdr_data, colWidths=[W])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ("SPAN",          (0,0), (-1,-1)),
    ]))
    story += [hdr, Spacer(1, 4*mm)]

    # Session meta
    story.append(Paragraph("Session Details",
                            S("H", fontName="Helvetica-Bold", fontSize=12,
                              textColor=NAVY, spaceBefore=6, spaceAfter=3)))
    meta_rows = [
        ["Session ID", session.session_id[:16] + "…"],
        ["Query",      session.query_echo[:80] + ("…" if len(session.query_echo) > 80 else "")],
        ["Generated",  session.timestamp],
    ]
    if session.data_warning:
        meta_rows.append(["Note", session.data_warning[:120]])
    if session.hec_eligibility and not session.hec_eligibility.get("eligible", True):
        meta_rows.append(["HEC Check", "⚠ Below threshold — see alternatives"])

    meta_tbl = Table(
        [[Paragraph(k, S("ML", fontName="Helvetica-Bold", fontSize=9, textColor=GREY)),
          Paragraph(v, S("MV", fontName="Helvetica", fontSize=9, textColor=NAVY))]
         for k, v in meta_rows],
        colWidths=[35*mm, W - 35*mm],
    )
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), LITE),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("LINEBELOW",     (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
    ]))
    story += [meta_tbl, Spacer(1, 5*mm)]

    # Recommendations
    recs   = session.response.data.recommendations
    badges = ["#FFD700", "#C0C0C0", "#CD7F32"]

    for idx, uni in enumerate(recs):
        badge_col = colors.HexColor(badges[idx] if idx < 3 else "#888888")

        # University header
        hdr_row = [[
            Paragraph(f"#{idx+1}  {uni.name}",
                      S("UN", fontName="Helvetica-Bold", fontSize=11,
                        textColor=colors.white)),
            Paragraph(f"{uni.city} · {uni.type} · HEC {uni.hec_category}",
                      S("UM", fontName="Helvetica", fontSize=9,
                        textColor=GOLD)),
        ]]
        uni_hdr = Table(hdr_row, colWidths=[W])
        uni_hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), NAVY),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        story.append(uni_hdr)

        # XAI explanation
        story.append(Paragraph(
            uni.xai_explanation,
            S("XAI", fontName="Helvetica-Oblique", fontSize=9,
              textColor=colors.HexColor("#222222"),
              leftIndent=6, rightIndent=6, spaceBefore=3, spaceAfter=3, leading=13),
        ))

        # Metrics row
        m     = uni.metrics
        m_row = [[
            Paragraph(f"Affordability: {m.affordability_score}/100",
                      S("M1", fontName="Helvetica", fontSize=9, textColor=NAVY)),
            Paragraph(f"Merit Chance: {m.merit_probability}%",
                      S("M2", fontName="Helvetica", fontSize=9, textColor=NAVY)),
            Paragraph(f"Market Value: {m.market_value}/100",
                      S("M3", fontName="Helvetica", fontSize=9, textColor=NAVY)),
        ]]
        m_tbl = Table(m_row, colWidths=[W/3, W/3, W/3])
        m_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), LITE),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ]))
        story.append(m_tbl)

        # Scholarships
        for sc in uni.scholarships_offered[:2]:
            story.append(Paragraph(
                f"Scholarship: {sc.name} — {sc.criteria} → {sc.coverage}",
                S("SC", fontName="Helvetica", fontSize=8.5,
                  textColor=TEAL, spaceBefore=2, leftIndent=6),
            ))

        # Links
        lnk_row = [[
            Paragraph(
                f'<link href="{uni.links.website}"><u>Website</u></link>',
                S("L1", fontName="Helvetica", fontSize=9, textColor=NAVY)),
            Paragraph(
                f'<link href="{uni.links.apply}"><u>Apply Now →</u></link>',
                S("L2", fontName="Helvetica-Bold", fontSize=9, textColor=GOLD)),
        ]]
        lnk_tbl = Table(lnk_row, colWidths=[W/2, W/2])
        lnk_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,0), LITE),
            ("BACKGROUND",    (1,0), (1,0), NAVY),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("ALIGN",         (1,0), (1,0), "CENTER"),
        ]))
        story += [lnk_tbl, Spacer(1, 5*mm)]

    # Footer
    story.append(HRFlowable(width=W, thickness=1, color=NAVY,
                             spaceBefore=4, spaceAfter=4))
    ftr = Table([[Paragraph(
        "Tahqiq.ai · Data ke paas jawab hain. Tahqiq.ai poochna jaanta hai. "
        "All data sourced from HEC Pakistan public records.",
        S("F", fontName="Helvetica-Oblique", fontSize=8,
          textColor=colors.white, alignment=TA_CENTER),
    )]], colWidths=[W])
    ftr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(ftr)

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI lifespan — wire router + seed KB
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_import_modules()
    if _AGENT_AVAILABLE:
        try:
            from llm_router import patch_agent_logic, get_router_status
            patch_agent_logic()
            status = get_router_status()
            print(f"[Tahqiq.ai] LLM router → active='{status['active_provider']}'  "
                  f"order={' → '.join(status['provider_order'])}")
        except Exception as e:
            print(f"[Tahqiq.ai] LLM router patch failed: {e}")

    if _KB_AVAILABLE:
        try:
            from knowledge_base import seed_if_empty
            result = seed_if_empty()
            print(f"[Tahqiq.ai] ChromaDB KB ready: {result}")
        except Exception as e:
            print(f"[Tahqiq.ai] ChromaDB seed failed (non-fatal): {e}")

    yield


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Tahqiq.ai API",
    description = "Har Student Ka Apna University Guide — HEC Gen AI Hackathon 2025",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten to frontend origin in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Health / utility endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    """Healthcheck — required by Hugging Face Spaces."""
    return {
        "status":          "ok",
        "service":         "Tahqiq.ai",
        "tagline":         "Har Student Ka Apna University Guide",
        "version":         "2.0.0",
        "agent_available": _AGENT_AVAILABLE,
        "kb_available":    _KB_AVAILABLE,
    }


@app.get("/kb/stats", tags=["Knowledge Base"])
def kb_stats():
    if not _KB_AVAILABLE:
        raise HTTPException(503, "Knowledge base not available.")
    from knowledge_base import get_collection_stats
    return get_collection_stats()


@app.get("/llm/status", tags=["LLM Router"])
def llm_status():
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "LLM router not available.")
    from llm_router import get_router_status
    return get_router_status()


@app.post("/llm/reset/{provider_name}", tags=["LLM Router"])
def llm_reset(provider_name: str):
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "LLM router not available.")
    from llm_router import reset_provider_health
    name = None if provider_name == "all" else provider_name
    return reset_provider_health(name)


@app.post("/kb/reset", tags=["Knowledge Base"])
def kb_reset():
    if not _KB_AVAILABLE:
        raise HTTPException(503, "Knowledge base not available.")
    from knowledge_base import reset_collection
    return reset_collection()


# ══════════════════════════════════════════════════════════════════════════════
# Core query endpoint — JSON only
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/query", response_model=QueryResponse, tags=["AI Query"])
def query(profile: StudentProfile):
    """
    Submit a student Urdish query → ranked university recommendations.

    Pipeline:
      1. Rule-based + LLM intent extraction (Urdish/English).
      2. HEC 45% threshold check.
      3. CSV-backed university retrieval + scoring.
      4. Grok-powered XAI explanation (Urdish).
      5. Next steps generation.
      6. Strict ContractResponse serialisation.
    """
    session_id = profile.session_id or str(uuid.uuid4())

    try:
        if _AGENT_AVAILABLE:
            from agent_logic import run_agent
            result = run_agent(
                raw_query      = profile.query,
                merit_override = profile.percentage,
            )
            contract     = result.contract
            data_warning = result.data_warning
            hec_check    = result.hec_eligibility
        else:
            contract     = _build_mock_response(profile)
            data_warning = "Agent unavailable — mock response returned."
            hec_check    = None

    except Exception as exc:
        raise HTTPException(
            status_code = 500,
            detail      = f"Agent pipeline error: {exc}",
        )

    out = QueryResponse(
        session_id      = session_id,
        query_echo      = profile.query,
        timestamp       = datetime.utcnow().isoformat(),
        response        = contract,
        data_warning    = data_warning,
        hec_eligibility = hec_check,
    )
    session_store[session_id] = out
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Multimodal endpoint — image upload + optional text query
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/query/multimodal", response_model=QueryResponse, tags=["AI Query"])
async def query_multimodal(
    query:      str            = Form(default="Kaunsi university mujhe best suit kare?"),
    percentage: Optional[float]= Form(default=None),
    field:      Optional[str]  = Form(default=None),
    city_pref:  Optional[str]  = Form(default=None),
    budget_pkr: Optional[int]  = Form(default=None),
    session_id: Optional[str]  = Form(default=None),
    image:      Optional[UploadFile] = File(default=None),
):
    """
    Multimodal endpoint: accepts an optional result card image + text query.

    If image is provided:
      1. Vision model (Grok-2-vision) extracts marks from the image.
      2. Extracted percentage overrides the percentage form field.
      3. Pipeline continues as normal.

    If image is absent, behaves identically to POST /query.
    """
    sid     = session_id or str(uuid.uuid4())
    ocr_res = None

    # ── OCR step ─────────────────────────────────────────────────────────────
    merit_from_ocr: Optional[float] = None
    if image is not None:
        try:
            raw_bytes  = await image.read()
            mime_type  = image.content_type or "image/jpeg"
            image_b64  = base64.b64encode(raw_bytes).decode("utf-8")

            if _AGENT_AVAILABLE:
                from agent_logic import extract_marks_from_image
                ocr_res = extract_marks_from_image(image_b64, mime_type)
                merit_from_ocr = ocr_res.get("marks_percent")
                if merit_from_ocr:
                    # Append OCR result to query for richer context
                    query = query + f" [OCR extracted marks: {merit_from_ocr}%]"
        except Exception as exc:
            # OCR failure is non-fatal — continue with text query
            ocr_res = {"error": str(exc), "marks_percent": None}

    # Determine final merit percent
    final_merit = merit_from_ocr if merit_from_ocr is not None else percentage

    # ── Agent pipeline ────────────────────────────────────────────────────────
    try:
        if _AGENT_AVAILABLE:
            from agent_logic import run_agent
            result = run_agent(raw_query=query, merit_override=final_merit)
            contract     = result.contract
            data_warning = result.data_warning
            hec_check    = result.hec_eligibility
        else:
            profile      = StudentProfile(
                query=query, percentage=final_merit,
                field=field, city_pref=city_pref, budget_pkr=budget_pkr,
            )
            contract     = _build_mock_response(profile)
            data_warning = "Agent unavailable — mock response returned."
            hec_check    = None

    except Exception as exc:
        raise HTTPException(
            status_code = 500,
            detail      = f"Multimodal pipeline error: {exc}",
        )

    out = QueryResponse(
        session_id      = sid,
        query_echo      = query,
        timestamp       = datetime.utcnow().isoformat(),
        response        = contract,
        data_warning    = data_warning,
        hec_eligibility = hec_check,
        ocr_result      = ocr_res,
    )
    session_store[sid] = out
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PDF download
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/download-report/{session_id}", tags=["Report"])
def download_report(session_id: str):
    """Generate and stream a branded PDF for a previous /query session."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(
            404,
            detail=f"Session '{session_id}' not found. Call /query first.",
        )
    try:
        pdf   = _generate_pdf(session)
        fname = f"tahqiq_report_{session_id[:8]}.pdf"
        return StreamingResponse(
            io.BytesIO(pdf),
            media_type = "application/pdf",
            headers    = {"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    except Exception as exc:
        raise HTTPException(500, detail=f"PDF generation error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Mock fallback (used when agent_logic not available)
# ══════════════════════════════════════════════════════════════════════════════

def _build_mock_response(profile: StudentProfile) -> ContractResponse:
    field    = profile.field or "Computer Science"
    city     = profile.city_pref or "Islamabad"
    pct      = profile.percentage or 78

    return ContractResponse(
        status = "success",
        data   = RecommendationsData(recommendations=[
            UniversityRecommendation(
                university_id        = "U001",
                name                 = "National University of Sciences and Technology (NUST)",
                city                 = "Islamabad",
                type                 = "Public",
                hec_category         = "W1",
                xai_explanation      = (
                    f"NUST Pakistan ka top W1 public university hai. "
                    f"Aapke {pct}% ke saath merit scholarship milne ki "
                    f"probability hai. {field} ke liye 42 PhD faculty members hain."
                ),
                scholarships_offered = [Scholarship(
                    name="NUST Merit Scholarship",
                    criteria="Top 10 in NET / CGPA 3.5+",
                    coverage="100% Tuition",
                )],
                metrics  = Metrics(affordability_score=60, merit_probability=40, market_value=98),
                links    = Links(website="https://nust.edu.pk",
                                 apply="https://nust.edu.pk/admissions/"),
            ),
            UniversityRecommendation(
                university_id        = "U008",
                name                 = "FAST-NUCES",
                city                 = city,
                type                 = "Private Non-Profit",
                hec_category         = "W2",
                xai_explanation      = (
                    f"FAST-NUCES {field} ke liye Pakistan ka best specialized university hai. "
                    f"Industry links strong hain, placement record 85%+ hai."
                ),
                scholarships_offered = [Scholarship(
                    name="FAST Merit Scholarship",
                    criteria="85%+ FSc",
                    coverage="50% Tuition",
                )],
                metrics  = Metrics(affordability_score=55, merit_probability=65, market_value=90),
                links    = Links(website="https://nu.edu.pk",
                                 apply="https://nu.edu.pk/admissions/"),
            ),
            UniversityRecommendation(
                university_id        = "U004",
                name                 = "COMSATS University Islamabad",
                city                 = "Islamabad",
                type                 = "Public",
                hec_category         = "W2",
                xai_explanation      = (
                    f"COMSATS affordable aur strong W2 university hai. "
                    f"Fee public rate par hai. {pct}% ke saath scholarship bhi possible hai."
                ),
                scholarships_offered = [Scholarship(
                    name="COMSATS Merit Award",
                    criteria="78%+ FSc",
                    coverage="50% Tuition Waiver",
                )],
                metrics  = Metrics(affordability_score=78, merit_probability=72, market_value=84),
                links    = Links(website="https://comsats.edu.pk",
                                 apply="https://admission.comsats.edu.pk/"),
            ),
        ]),
    )
