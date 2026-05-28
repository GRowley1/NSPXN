from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any
import os
import re
import glob

from fastapi import FastAPI, UploadFile, File, Form, Response, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fpdf import FPDF

PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

# Optional report logo. In Render, set NSPXN_LOGO_PATH if the file is stored elsewhere.
NSPXN_LOGO_PATH = os.getenv("NSPXN_LOGO_PATH", os.path.join(os.path.dirname(__file__), "logo2.png"))

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


def _looks_like_nspxn_user_id(value: Any) -> bool:
    return bool(re.fullmatch(r"(?i)NSPXN\d+", str(value or "").strip()))


US_STATE_ABBRS = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD",
    "ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VA","VT","WA","WI","WV","WY","DC"
}

US_STATE_NAMES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO",
    "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
    "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC",
}

def _normalize_state_value(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or _looks_like_nspxn_user_id(raw):
        return ""

    s = re.sub(r"(?i)^\s*(?:state|loss\s*state|claim\s*state|vehicle\s*state)\s*[:=\-]\s*", "", raw).strip()
    up = re.sub(r"\s+", " ", s.upper()).strip()

    # Critical: do NOT allow Yes/No fields to become fake state codes.
    if up in {"N/A", "NA", "NONE", "NULL", "UNKNOWN", "SELECT", "SELECT STATE", "-- SELECT STATE --",
              "YES", "NO", "Y", "N", "TRUE", "FALSE", "ON", "OFF"}:
        return ""

    if re.fullmatch(r"[A-Z]{2}", up) and up in US_STATE_ABBRS:
        return up

    # Common select display values: "CO - Colorado", "CO/Colorado", "CO (Colorado)".
    m = re.search(r"\b([A-Z]{2})\b", up)
    if m and m.group(1) in US_STATE_ABBRS:
        return m.group(1)

    cleaned_name = re.sub(r"[^A-Z ]+", " ", up)
    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()
    if cleaned_name in US_STATE_NAMES:
        return US_STATE_NAMES[cleaned_name]
    for name, abbr in US_STATE_NAMES.items():
        if re.search(r"\b" + re.escape(name) + r"\b", cleaned_name):
            return abbr
    return ""


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

    # Header / branding
    pdf.set_fill_color(8, 12, 18)
    pdf.rect(0, 0, 210, 32, "F")
    logo_drawn = False
    try:
        if NSPXN_LOGO_PATH and os.path.exists(NSPXN_LOGO_PATH):
            pdf.image(NSPXN_LOGO_PATH, x=10, y=5, w=82)
            logo_drawn = True
    except Exception:
        logo_drawn = False
    if not logo_drawn:
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 18)
        pdf.set_xy(10, 8)
        pdf.cell(0, 8, "NSPXN.com", ln=True)

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 13)
    pdf.set_xy(100, 9)
    pdf.multi_cell(100, 6, title, align="R")
    pdf.ln(14)
    pdf.set_text_color(0, 0, 0)

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    section_colors = {
        "Preliminary Diminished Value Screening": (0, 132, 204),
        "Vehicle / Claim Inputs": (31, 42, 53),
        "17c Reference Calculation": (0, 132, 204),
        "Market-Based DV Screening Range": (31, 42, 53),
        "Screening Notes": (0, 132, 204),
        "Disclaimer": (96, 96, 96),
    }

    def _pdf_safe_text(value: str) -> str:
        text = str(value or "")
        text = text.replace("\t", "    ")
        text = text.replace("–", "-").replace("—", "-").replace("×", "x")
        text = text.replace("“", '"').replace("”", '"').replace("’", "'")
        text = re.sub(r"[^\S\r\n]+", " ", text)
        return text.encode("latin-1", "replace").decode("latin-1")

    def _draw_section_bar(label: str) -> None:
        color = section_colors.get(label, (0, 132, 204))
        pdf.ln(2)
        pdf.set_x(pdf.l_margin)
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(usable_width, 7, label.upper(), border=0, ln=True, align="L", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)
        pdf.ln(1)

    def _draw_key_value(line: str) -> None:
        if ":" not in line:
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(usable_width, 5, line)
            return
        key, val = line.split(":", 1)
        key = key.strip() + ":"
        val = val.strip()
        key_w = 48
        if pdf.get_y() > 270:
            pdf.add_page()
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Arial", "B", 9.5)
        pdf.cell(key_w, 5.5, key, border=0)
        pdf.set_font("Arial", "", 9.5)
        pdf.multi_cell(usable_width - key_w, 5.5, val)

    for raw_line in lines:
        safe_line = _pdf_safe_text(raw_line).strip()
        if safe_line.startswith("## "):
            label = safe_line.replace("## ", "", 1).strip()
            _draw_section_bar(label)
        elif safe_line == "":
            pdf.ln(2)
        else:
            pdf.set_x(pdf.l_margin)
            if safe_line.startswith("This is a Preliminary Diminished Value Screening only"):
                pdf.set_text_color(110, 110, 110)
                pdf.set_font("Arial", "", 6)
                pdf.multi_cell(usable_width, 4, safe_line)
                pdf.set_text_color(0, 0, 0)
            elif ":" in safe_line and len(safe_line.split(":", 1)[0]) <= 36:
                _draw_key_value(safe_line)
            else:
                pdf.set_font("Arial", "", 7)
                pdf.multi_cell(usable_width, 5, safe_line)

    pdf.output(pdf_path)
    return pdf_filename


@app.post("/vision-review")
async def vision_review(
    request: Request,
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

    # Resolve DV state defensively. Never guess or force a default state.
    # If the frontend selected CO/TN/etc. but the value does not reach this backend,
    # block instead of producing a customer PDF with the wrong state.
    resolved_state = _normalize_state_value(dv_state)
    _safe_form_keys: List[str] = []
    if not resolved_state or _looks_like_nspxn_user_id(dv_state):
        try:
            form = await request.form()

            # Known frontend aliases first.
            for key in (
                "dv_state", "dv-state", "dvState", "dv_state_input", "dvStateInput",
                "dv_state_select", "dvStateSelect", "dv_state_value", "dvStateValue",
                "dv_claim_state", "dvClaimState", "dv_loss_state", "dvLossState",
                "dv-loss-state", "dv-claim-state", "state", "state_select", "stateSelect",
                "claim_state", "claimState", "claim_state_select", "claimStateSelect",
                "loss_state", "lossState", "loss_state_select", "lossStateSelect",
                "loss_location_state", "lossLocationState", "vehicle_state", "vehicleState",
                "jurisdiction_state", "jurisdictionState", "dv_jurisdiction", "dvJurisdiction",
                "selected_state", "selectedState", "selected-state", "state_selected", "stateSelected",
                "dv_selected_state", "dvSelectedState", "dv-state-selected", "dvStateSelected",
                "claim_jurisdiction", "claimJurisdiction", "claim-jurisdiction",
            ):
                candidate = _normalize_state_value(form.get(key, ""))
                if candidate:
                    resolved_state = candidate
                    break

            # Then any key containing state/jurisdiction/location.
            if not resolved_state:
                for key in form.keys():
                    try:
                        value = form.get(key, "")
                        if hasattr(value, "filename"):
                            continue
                        _safe_form_keys.append(str(key))
                        key_l = str(key or "").lower()
                        if not any(token in key_l for token in ("state", "jurisdiction", "location")):
                            continue
                        candidate = _normalize_state_value(value)
                        if candidate:
                            resolved_state = candidate
                            break
                    except Exception:
                        continue

        except Exception:
            pass

    if not resolved_state:
        return JSONResponse(
            status_code=200,
            content={
                "status": "blocked",
                "error": "Missing required DV field: State",
                "detail": "A valid two-letter state was not received by the diminished value backend. The report was blocked so it does not print N/A or the wrong state.",
                "file_number": file_number,
                "request_type": "Preliminary Diminished Value Screening",
                "safe_form_keys_received": _safe_form_keys[:80],
            },
        )

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
        f"State: {resolved_state}",
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
        "Strict 17c may calculate $0 when mileage exceeds 100,000 miles because the 17c mileage modifier becomes 0.00 at that threshold. This does not mean diminished value exposure is necessarily zero. The Market-Based DV Screening Range is provided separately to reflect potential market stigma, repair severity, repair-to-value ratio, claim type, and vehicle-specific loss factors. This screening uses user-provided pre-loss value, mileage, repair total, damage severity, and known claim indicators. The 17c figure is provided only as a reference calculation and may be lower than market-based diminished value. The market-based range is a preliminary screening position and should be supported with comparable market data when used in a formal claim presentation.",
        "",
        "## Disclaimer",
        "This is a Preliminary Diminished Value Screening only. It is not a certified diminished value appraisal unless reviewed, finalized, and signed by a qualified appraiser. NSPXN does not independently retrieve third-party valuation data; the pre-loss value is based on user-provided information and should be verified against accepted market valuation sources.",
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
