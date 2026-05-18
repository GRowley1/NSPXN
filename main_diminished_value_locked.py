from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any
import os
import re
import glob

from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fpdf import FPDF

PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com", "https://www.nspxn.com", "http://nspxn.com", "http://www.nspxn.com",
        "https://nspxn.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "-", (s or "").strip()).strip("-_. ")


def _money(v: Optional[float]) -> str:
    try:
        return "${:,.0f}".format(float(v or 0))
    except Exception:
        return "$0"


def _num(v: Any) -> float:
    if v is None:
        return 0.0
    s = str(v).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def _clean(v: Any, default: str = "N/A") -> str:
    s = str(v or "").strip()
    return s if s else default


def _mileage_multiplier(mileage: float) -> float:
    if mileage < 20000:
        return 1.00
    if mileage < 40000:
        return 0.80
    if mileage < 60000:
        return 0.60
    if mileage < 80000:
        return 0.40
    if mileage < 100000:
        return 0.20
    return 0.00


def _damage_multiplier(severity: str, structural_damage: str, airbag_deployment: str) -> float:
    sev = (severity or "").strip().lower()
    structural = (structural_damage or "").strip().lower()
    airbag = (airbag_deployment or "").strip().lower()
    if structural == "yes" or airbag == "yes" or "severe" in sev or "major" in sev:
        return 1.00 if (structural == "yes" or airbag == "yes" or "severe" in sev) else 0.75
    if "moderate" in sev:
        return 0.50
    if "minor" in sev:
        return 0.25
    return 0.50


def _market_range_percent(severity: str, structural_damage: str, airbag_deployment: str, prior_accident_history: str) -> tuple[float, float, str]:
    sev = (severity or "").strip().lower()
    structural = (structural_damage or "").strip().lower()
    airbag = (airbag_deployment or "").strip().lower()
    prior = (prior_accident_history or "").strip().lower()

    if structural == "yes" or airbag == "yes" or "severe" in sev:
        low, high, label = 0.15, 0.25, "High DV exposure"
    elif "major" in sev:
        low, high, label = 0.10, 0.18, "Elevated DV exposure"
    elif "moderate" in sev:
        low, high, label = 0.05, 0.10, "Moderate DV exposure"
    elif "minor" in sev:
        low, high, label = 0.02, 0.05, "Low DV exposure"
    else:
        low, high, label = 0.04, 0.08, "Unconfirmed DV exposure"

    if prior == "yes":
        low *= 0.60
        high *= 0.60
        label += "; reduced because prior accident history was reported"
    return low, high, label


def _write_pdf(file_number: str, title: str, lines: List[str]) -> str:
    safe_file = _safe(file_number or "dv-screening") or "dv-screening"
    pdf_filename = f"{safe_file}.pdf"
    pdf_path = os.path.join(PDF_DIR, pdf_filename)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(12, 18, 26)
    pdf.rect(0, 0, 210, 24, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "NSPXN.com", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, title, ln=True)
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)

    for line in lines:
        if line.startswith("## "):
            pdf.ln(2)
            pdf.set_fill_color(230, 230, 230)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, line.replace("## ", ""), ln=True, fill=True)
            pdf.set_font("Arial", "", 10)
        elif line.strip() == "":
            pdf.ln(3)
        else:
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, line.encode("latin-1", "replace").decode("latin-1"))

    pdf.output(pdf_path)
    return pdf_filename


@app.post("/vision-review")
async def vision_review(
    response: Response,
    files: Optional[List[UploadFile]] = File(None),
    file_number: Optional[str] = Form(None),
    ia_company: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("preliminary_diminished_value_screening"),
    dv_vin: str = Form(""),
    dv_year_make_model: str = Form(""),
    dv_mileage: str = Form(""),
    dv_pre_loss_value: str = Form(""),
    dv_pre_loss_value_source: str = Form(""),
    dv_repair_total: str = Form(""),
    dv_damage_severity: str = Form(""),
    dv_prior_accident_history: str = Form("Unknown"),
    dv_structural_damage: str = Form("Unknown"),
    dv_airbag_deployment: str = Form("Unknown"),
    dv_claim_type: str = Form("Unknown"),
    dv_state: str = Form(""),
):
    if (ai_intent or "").strip().lower() != "preliminary_diminished_value_screening":
        return JSONResponse(status_code=400, content={"error": "Invalid DV request type."})

    if not file_number or not str(file_number).strip():
        return JSONResponse(status_code=400, content={"error": "Missing required field: file_number"})

    pre_loss_value = _num(dv_pre_loss_value)
    mileage = _num(dv_mileage)
    repair_total = _num(dv_repair_total)

    missing = []
    if pre_loss_value <= 0:
        missing.append("Pre-Loss Value")
    if mileage <= 0:
        missing.append("Mileage")
    if repair_total <= 0:
        missing.append("Repair Total")
    if missing:
        return JSONResponse(status_code=400, content={"error": "Missing required DV field(s): " + ", ".join(missing)})

    damage_mult = _damage_multiplier(dv_damage_severity, dv_structural_damage, dv_airbag_deployment)
    mileage_mult = _mileage_multiplier(mileage)
    base_loss = pre_loss_value * 0.10
    dv_17c = base_loss * damage_mult * mileage_mult

    low_pct, high_pct, exposure_label = _market_range_percent(
        dv_damage_severity, dv_structural_damage, dv_airbag_deployment, dv_prior_accident_history
    )
    market_low = pre_loss_value * low_pct
    market_high = pre_loss_value * high_pct
    recommended = (market_low + market_high) / 2.0

    repair_ratio = repair_total / pre_loss_value if pre_loss_value else 0.0
    generated = datetime.now(ZoneInfo("America/New_York")).strftime("%m/%d/%Y %I:%M %p EST")

    markdown_lines = [
        "## Preliminary Diminished Value Screening",
        f"Generated: {generated}",
        f"File Number: {_clean(file_number)}",
        f"Inspected For: {_clean(ia_company)}",
        f"NSPXN User ID #: {_clean(appraiser_id)}",
        "",
        "## Vehicle / Claim Inputs",
        f"VIN: {_clean(dv_vin)}",
        f"Vehicle: {_clean(dv_year_make_model)}",
        f"Mileage: {mileage:,.0f}",
        f"State: {_clean(dv_state)}",
        f"Claim Type: {_clean(dv_claim_type)}",
        f"Pre-Loss Value: {_money(pre_loss_value)}",
        f"Pre-Loss Value Source: {_clean(dv_pre_loss_value_source)}",
        f"Repair Total: {_money(repair_total)}",
        f"Repair-to-Value Ratio: {repair_ratio:.1%}",
        f"Damage Severity: {_clean(dv_damage_severity)}",
        f"Structural Damage: {_clean(dv_structural_damage)}",
        f"Airbag Deployment: {_clean(dv_airbag_deployment)}",
        f"Prior Accident History: {_clean(dv_prior_accident_history)}",
        "",
        "## 17c Reference Calculation",
        f"Base Loss of Value Cap: {_money(pre_loss_value)} x 10% = {_money(base_loss)}",
        f"Damage Multiplier: {damage_mult:.2f}",
        f"Mileage Multiplier: {mileage_mult:.2f}",
        f"17c Reference DV: {_money(dv_17c)}",
        "",
        "## Market-Based DV Screening Range",
        f"Exposure Category: {exposure_label}",
        f"Market-Based DV Range: {low_pct:.0%} - {high_pct:.0%} of pre-loss value",
        f"Low Range: {_money(market_low)}",
        f"High Range: {_money(market_high)}",
        f"Preliminary Review Position: {_money(recommended)}",
        "",
        "## Screening Notes",
        "This screening uses user-provided pre-loss value, mileage, repair total, damage severity, and known claim indicators. The 17c figure is provided only as a reference calculation and may be lower than market-based diminished value. The market-based range is a preliminary screening position and should be supported with comparable market data when used in a formal claim presentation.",
        "",
        "## Disclaimer",
        "This is a Preliminary Diminished Value Screening only. It is not a certified diminished value appraisal unless reviewed, finalized, and signed by a qualified appraiser. NSPXN does not auto-pull KBB, NADA, J.D. Power, Black Book, or dealer valuation data in this Phase 1 workflow; the pre-loss value is user-provided.",
    ]
    summary_markdown = "\n".join(markdown_lines)
    pdf_filename = _write_pdf(str(file_number), "Preliminary Diminished Value Screening", markdown_lines)

    response.headers["X-NSPXN-Report-Completed"] = "true"
    response.headers["X-NSPXN-AI-Intent"] = "preliminary_diminished_value_screening"
    response.headers["X-NSPXN-File-Number"] = str(file_number or "")

    return {
        "file_number": file_number,
        "request_type": "Preliminary Diminished Value Screening",
        "summary_brief": f"Preliminary DV range: {_money(market_low)} - {_money(market_high)}. 17c reference: {_money(dv_17c)}.",
        "summary_markdown": summary_markdown,
        "gpt_output": summary_markdown,
        "pdf_url": f"/download-pdf?filename={pdf_filename}",
        "pdf_filename": pdf_filename,
        "dv_result": {
            "pre_loss_value": pre_loss_value,
            "mileage": mileage,
            "repair_total": repair_total,
            "damage_multiplier": damage_mult,
            "mileage_multiplier": mileage_mult,
            "dv_17c": dv_17c,
            "market_low": market_low,
            "market_high": market_high,
            "recommended_position": recommended,
        },
    }


@app.get("/download-pdf")
async def download_pdf(file_number: Optional[str] = None, filename: Optional[str] = None):
    if filename:
        safe = _safe(filename)
        path = os.path.join(PDF_DIR, safe)
        if os.path.exists(path):
            return FileResponse(path=path, media_type="application/pdf", filename=safe)
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    if not file_number:
        return JSONResponse(status_code=400, content={"detail": "Missing query param 'filename' or 'file_number'"})
    safe_num = _safe(file_number)
    candidates = glob.glob(os.path.join(PDF_DIR, f"*{safe_num}*.pdf"))
    if not candidates:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    latest = max(candidates, key=lambda p: os.path.getmtime(p))
    return FileResponse(path=latest, media_type="application/pdf", filename=os.path.basename(latest))
