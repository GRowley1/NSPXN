
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
import os, io, re, time, base64, zipfile, smtplib, json
from email.message import EmailMessage

from fpdf import FPDF
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

from openai import OpenAI

# ======================= Config =======================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

MODEL_PRIMARY = os.getenv("OAI_MODEL", "gpt-4o-mini")
MODEL_FALLBACK = "gpt-3.5-turbo"

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")
# Avoid passing unsupported kwargs to client constructor
client = OpenAI(api_key=api_key)

INTENTS = {
    "guidelines_only": "Guidelines → Estimate (no photos)",
    "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
    "photos_only": "Photos Only: Compare to Estimate",
    "invoices_with_photos": "Supplement ↔ Invoices (+ Photos)",
    "docs_checklist": "Documentation Checklist",
}

# ======================= App =======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= Utilities (lean) =======================
def _pp(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.7)
    img = ImageOps.autocontrast(img)
    return img

def sanitize_latin1(s: str) -> str:
    if not s: return ""
    repl = {
        "\u2018":"'",
        "\u2019":"'",
        "\u201C":'"',
        "\u201D":'"',
        "\u2013":"-",
        "\u2014":"-",
        "\u2022":"-",
        "\u2026":"...",
        "\u2192":"->",
        "\u2194":"<->",
        "\u00A0":" ",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "ignore").decode("latin-1", "ignore")

def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.\-]+", "-", s)
    return s.strip("-_.") or f"report-{int(time.time())}"

def fast_pdf_text(pdf_bytes: bytes, limit_pages: int = 12) -> str:
    out = []
    try:
        rdr = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for i, pg in enumerate(rdr.pages[:limit_pages], 1):
            t = pg.extract_text() or ""
            if t.strip():
                out.append(f"[Page {i}]\n{t}")
    except Exception:
        pass
    return "\n\n".join(out)

def first_page_ocr(pdf_bytes: bytes, dpi: int = 300) -> str:
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)[:1]
    except Exception:
        return ""
    if not imgs:
        return ""
    try:
        return "[OCR 1]\n" + pytesseract.image_to_string(_pp(imgs[0]), lang="eng", config="--psm 6")
    except Exception:
        return ""

def ocr_pages_for_vin(pdf_bytes: bytes, max_pages: int = 4, dpi: int = 250) -> str:
    """VIN scan helper (estimate only, small cap to keep latency down)."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
    except Exception:
        return ""
    out = []
    for im in pages:
        try:
            im = _pp(im)
            out.append(pytesseract.image_to_string(im, lang="eng", config="--psm 6"))
        except Exception:
            pass
    return "\n".join(out)

# ---------- Light extraction (VIN / Claim / Vehicle / Odometer / Days) ----------
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
VIN_TIGHT = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
VIN_RELAX = re.compile(
    r"(?:V\.?I\.?N\.?|Vehicle\s+Identification\s+Number|VIN)\b[^A-Z0-9]{0,40}((?:[A-HJ-NPR-Z0-9][\s\-]*){17})",
    re.IGNORECASE,
)
_trans = {
    **{str(i): i for i in range(10)},
    **dict(A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8, J=1, K=2, L=3, M=4, N=5, P=7, R=9, S=2, T=3, U=4, V=5, W=6, X=7, Y=8, Z=9),
}
_w = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def _norm_vin(s: str) -> Optional[str]:
    s = (s or "").upper()
    s = re.sub(r"[^A-HJ-NPR-Z0-9]", "", s).replace("O", "0").replace("I", "1").replace("Q", "0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s):
        return None
    return s

def _vin_ok(v: str) -> bool:
    try:
        tot = sum(_trans[ch]*_w[i] for i, ch in enumerate(v))
        chk = tot % 11
        return v[8] == ("X" if chk == 10 else str(chk))
    except Exception:
        return False

def vin_from_text(text: str) -> Optional[str]:
    cands = [m.group(1) for m in VIN_RELAX.finditer(text or "")] + VIN_TIGHT.findall(text or "")
    uniq, seen = [], set()
    for c in cands:
        v = _norm_vin(c)
        if v and v not in seen:
            uniq.append(v)
            seen.add(v)
    for v in uniq:
        if _vin_ok(v):
            return v
    return uniq[0] if uniq else None

def scan_estimate_for_vin(pdf_bytes: bytes) -> Optional[str]:
    """Robust estimate VIN: text (≤12 pages) then OCR (≤4 pages); prefer check-digit valid; else first normalized 17-char."""
    text_12 = fast_pdf_text(pdf_bytes, limit_pages=12)
    v = vin_from_text(text_12)
    if v:
        return v
    ocr_text = ocr_pages_for_vin(pdf_bytes, max_pages=4, dpi=250)
    v = vin_from_text(text_12 + "\n" + ocr_text)
    if v:
        return v
    for c in re.findall(r"\b([A-HJ-NPR-Z0-9\-\s]{17,40})\b", text_12 + "\n" + ocr_text, flags=re.I):
        cc = re.sub(r"[^A-HJ-NPR-Z0-9]", "", c.upper()).replace("O", "0").replace("I", "1").replace("Q", "0")
        if len(cc) == 17 and re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", cc):
            return cc
    return None

MAKE_MAP = {
    "NISSAN":"Nissan","CHEV":"Chevrolet","CHEVY":"Chevrolet","TOYOTA":"Toyota","FORD":"Ford","HONDA":"Honda",
    "HYUNDAI":"Hyundai","KIA":"Kia","BMW":"BMW","MERCEDES":"Mercedes-Benz","MB":"Mercedes-Benz",
    "VW":"Volkswagen","VOLKS":"Volkswagen","SUBARU":"Subaru","MAZDA":"Mazda","DODGE":"Dodge"
}
STOP = {
    "GASOLINE","DIESEL","HYBRID","ELECTRIC","BLACK","WHITE","BLUE","RED","SILVER","GRAY","GREY",
    "4D","2D","SED","SDN","SUV","COUPE","HATCH","TRUCK","WAGON","AWD","FWD","RWD","L","GDI","TURBO","PAINT","CLEAR","COAT","COLOR"
}

def vehicle_from_text(text: str) -> Optional[str]:
    for ln in (text or "").splitlines():
        ln = re.sub(r"\s{2,}", " ", ln.strip())
        if re.match(r"^\s*(19|20)\d{2}\b", ln) and not re.search(r"\b(AM|PM)\b", ln):
            parts = ln.split()
            year = parts[0]
            tail = parts[1:]
            keep = []
            for t in tail:
                raw = re.sub(r"[^\w\-]", "", t).upper()
                if raw in STOP or raw in ("A/M","OEM"):
                    break
                keep.append(t)
                if len(keep) >= 4:
                    break
            if keep:
                mk = MAKE_MAP.get(keep[0].upper(), keep[0].capitalize())
                return " ".join([year, mk] + keep[1:])
    return None

def mileage_from_text(text: str) -> Optional[str]:
    m = re.search(r"(?:Odometer|Mileage|Miles)\s*[:\-]?\s*([\d,]{2,7})\b", text or "", re.IGNORECASE)
    return m.group(1) if m else None

MODEL_WORDS = re.compile(r"(sedan|coupe|hatch|suv|truck|van|wagon|convertible|corolla|altima|civic|accord|camry|sentra|elantra|rogue|rav4|tacoma|silverado|ram|f150)", re.I)

def _is_plausible_claim(tok: str) -> bool:
    if not tok:
        return False
    tok = tok.strip().strip(".:,;#")
    tok = re.sub(r"\s+", "", tok)
    if not (5 <= len(tok) <= 30):
        return False
    if len(re.findall(r"\d", tok)) < 2:
        return False
    if "/" in tok and MODEL_WORDS.search(tok):
        return False
    if re.match(r"\d{4}-\d{2}-\d{2}$", tok):
        return False
    return True

def claim_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    flat = re.sub(r"[\r\n]+", " ", text)
    flat = re.sub(r"\s{2,}", " ", flat)
    LABELS = r"(?:Claim|Assignment|Reference|Ref|File|Loss|Case|Report|RO|Work\s*Order)"
    TOKEN  = r"([A-Za-z0-9][A-Za-z0-9\-_\/]*\d[A-Za-z0-9\-_\/]*)"
    for pat in [
        rf"{LABELS}\s*(?:No\.?|Number|#)?\s*[:\-]?\s*{TOKEN}",
        rf"{LABELS}\s*[:\-]?\s*(?:No\.?|Number|#)?\s*{TOKEN}",
    ]:
        m = re.search(pat, flat, re.IGNORECASE)
        if m:
            cand = m.group(1)
            if _is_plausible_claim(cand):
                return cand.strip().rstrip(".:,;")
    for m in re.finditer(r"\b\d{4,}-\d{5,}\b", flat):
        start, end = m.start(), m.end()
        window = flat[max(0, start-80):min(len(flat), end+80)]
        if re.search(LABELS, window, re.IGNORECASE):
            cand = m.group(0)
            if _is_plausible_claim(cand):
                return cand
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-_\/]*\d[A-Za-z0-9\-_\/]*", flat):
        cand = m.group(0)
        if _is_plausible_claim(cand) and not MODEL_WORDS.search(cand):
            return cand
    return None

def days_from_text(text: str) -> Optional[int]:
    m = re.search(r"Days?\s*to\s*Repair\s*[:\-]?\s*([0-9]+)", text or "", re.IGNORECASE)
    try:
        return int(m.group(1)) if m else None
    except Exception:
        return None

def estimate_total_from_text(text: str) -> Optional[str]:
    """Extract estimate's printed total like '4,162.86' if present."""
    if not text:
        return None
    PAT = re.compile(
        r"(Grand\s*Total|Net\s*Total|Estimate\s*Total|Total\s*Repairs?|Repair\s*Total)\s*[:\-]?\s*\$?\s*([0-9][0-9,]*\.\d{2})",
        re.IGNORECASE,
    )
    m = PAT.search(text)
    return m.group(2) if m else None

def detect_valuations(pdfs: List[Tuple[str, bytes]], est_pdf_name: Optional[str]) -> Tuple[bool, list, list]:
    VAL_KEYS = [("Kelley Blue Book","KBB"),("KBB.com","KBB"),("NADA","NADA/J.D. Power"),("J.D. Power","NADA/J.D. Power"),("Black Book","Black Book"),("Edmunds","Edmunds")]
    GENERIC_PAT = re.compile(r"(Average Price|Estimated Trade-?In Value|Clean Retail Value|Vehicle Valuation)", re.I)
    sources, names = [], []
    for nm, blob in pdfs:
        if est_pdf_name and nm == est_pdf_name:
            continue
        try:
            t = (first_page_ocr(blob) + "\n" + fast_pdf_text(blob, limit_pages=2)).upper()
        except Exception:
            t = ""
        hit = None
        for key, lab in VAL_KEYS:
            if key.upper() in t:
                hit = lab
                break
        if hit is None and GENERIC_PAT.search(t):
            hit = "Generic Valuation"
        if hit:
            sources.append(hit)
            names.append(nm)
    sources = list(dict.fromkeys(sources))
    return (len(sources) > 0), sources, names

def collect_invoices(pdfs, est_pdf_name: Optional[str]):
    """Return list of (name, bytes, header_hit) for likely INVOICE/SUPPLEMENT docs (not the estimate)."""
    invoice_like = []
    FN_KEYS = ("invoice","inv ", "receipt","bill","supplement")
    TXT_KEYS = ("INVOICE","INVOICE #","INVOICE NO","RECEIPT","BILLING","SUPPLEMENT")
    for nm, blob in pdfs:
        if est_pdf_name and nm == est_pdf_name:
            continue
        lower = nm.lower()
        hit = None
        if any(k in lower for k in FN_KEYS):
            hit = "filename"
        else:
            header = (first_page_ocr(blob) + "\n" + fast_pdf_text(blob, limit_pages=1)).upper()
            if any(k in header for k in TXT_KEYS):
                hit = "header"
        if hit:
            invoice_like.append((nm, blob, hit))
    return invoice_like

def final_percent(narr: str) -> Optional[int]:
    m = re.search(r"(final\s*(evaluation|score|compliance)\s*[:\-]?\s*)(\d{1,3})\s*%", narr or "", re.IGNORECASE)
    if m:
        try:
            v = int(m.group(3))
            return v if 0 <= v <= 100 else None
        except Exception:
            return None
    return None

def parse_vin_verification(narr: str) -> Optional[str]:
    m = re.search(r"VIN\s*Verification\s*:\s*(MATCH|MISMATCH|NOT\s*VERIFIED|PHOTOS\s*NOT\s*PROVIDED)", narr or "", re.IGNORECASE)
    return m.group(1).upper().replace("  ", " ") if m else None

def openai_chat(messages, max_tokens=900):
    # Short retry; fall back to a lighter model
    for attempt in range(2):
        try:
            return client.chat.completions.create(
                model=MODEL_PRIMARY, messages=messages, max_tokens=max_tokens, temperature=0
            )
        except Exception as e:
            if "429" in str(e) or "timeout" in str(e).lower():
                time.sleep(1.25 * (attempt + 1))
                continue
            break
    try:
        return client.chat.completions.create(
            model=MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=0
        )
    except Exception:
        return None

# ======================= API =======================
@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company:: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("guidelines_only")
):
    intent = ai_intent if ai_intent in INTENTS else "guidelines_only"
    request_type_label = INTENTS[intent]
    file_number = safe_filename(file_number)

    # --- Gather files ---
    pdfs: List[Tuple[str, bytes]] = []
    images: List[Tuple[str, bytes]] = []
    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            pdfs.append((name, raw))
        elif name.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")):
            images.append((name, raw))
        elif name.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for zi in z.infolist():
                        if zi.is_dir():
                            continue
                        zname = zi.filename.lower()
                        zdata = z.read(zi)
                        if zname.endswith(".pdf"):
                            pdfs.append((zname, zdata))
                        elif zname.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")):
                            images.append((zname, zdata))
            except Exception:
                pass

    photos_present = len(images) > 0

    # --- Identify estimate PDF (estimate-first) ---
    est_pdf = None
    for nm, blob in pdfs:
        if "est" in nm or "estimate" in nm:
            est_pdf = (nm, blob)
            break
    if est_pdf is None and pdfs:
        est_pdf = pdfs[0]

    # --- Extract estimate text (lean + robust for VIN/Claim) ---
    est_text = ""
    if est_pdf:
        t12 = fast_pdf_text(est_pdf[1], limit_pages=12)
        est_text = (first_page_ocr(est_pdf[1]) + "\n\n" + t12).strip()
        # Ensure VIN captured even if not in text
        if not vin_from_text(est_text):
            est_text += "\n\n" + ocr_pages_for_vin(est_pdf[1], max_pages=4, dpi=250)

    # --- Aggregate light text from other PDFs (only small amount) ---
    all_pdf_text = est_text
    for nm, blob in pdfs:
        if not est_pdf or nm != est_pdf[0]:
            all_pdf_text += "\n\n" + first_page_ocr(blob) + "\n\n" + fast_pdf_text(blob, limit_pages=2)

    # --- Header fields ---
    vin_estimate = scan_estimate_for_vin(est_pdf[1]) if est_pdf else None
    vin = vin_estimate or vin_from_text(est_text) or vin_from_text(all_pdf_text) or "N/A"
    vehicle = vehicle_from_text(est_text) or vehicle_from_text(all_pdf_text) or "N/A"
    mileage = mileage_from_text(est_text) or mileage_from_text(all_pdf_text)
    claim = claim_from_text(est_text) or claim_from_text(all_pdf_text) or "N/A"
    days = days_from_text(est_text) or days_from_text(all_pdf_text)

    est_total = estimate_total_from_text(est_text)
    # Valuation docs (ANY source allowed)
    est_name = est_pdf[0] if est_pdf else None
    valuation_present, valuation_sources, valuation_docs = detect_valuations(pdfs, est_name)

    # --- Client rules: textarea only (never scrape from uploads) ---
    rules_text = (client_rules or "").strip()
    rules_present = len(rules_text) > 0

    # --- GPT prompt (GPT does the heavy lifting) ---
    sys_common = (
        "You are an auto-claims appraisal assistant. Use ONLY the provided materials (estimate text, uploaded photos, client rules textarea). "
        "Do NOT invent details; if not present, write 'Not found in provided documents'. "
        "Total-loss-only items must be 'Not applicable (repairable)' unless the estimate explicitly declares total loss. "
        "For 'Clean Retail Value', count ANY uploaded valuation (KBB, NADA/J.D. Power, Black Book, Edmunds, generic) as compliant; name any sources detected. "
        "Always end with a single line: Final Evaluation: NN%."
    )

    header_hint = {
        "photos_present": photos_present,
        "vin_estimate": vin if vin != "N/A" else "",
        "vehicle": vehicle,
        "claim": claim if claim != "N/A" else "",
        "mileage_estimate": mileage or "",
        "days_to_repair_estimate": days if days is not None else "",
        "valuation_present": valuation_present,
        "valuation_sources": valuation_sources,
        "estimate_total": est_total,
        "rules_present": rules_present,
        "intent": intent,
    }

    messages = [{"role": "system", "content": sys_common + " Parsed header context: " + json.dumps(header_hint)}]

    def img_payload(max_imgs=12):
        out = []
        for _, blob in images[:max_imgs]:
            b64 = base64.b64encode(blob).decode("utf-8")
            out.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return out

    if intent == "guidelines_only":
        system = (
            "Compare client guidelines to the ESTIMATE only. "
            "Do NOT restate guidelines verbatim. "
            "Output sections: Estimate Snapshot (only if present), Compliance vs Guidelines (Compliant/Non-compliant/Not found with short estimate quote), Missing Documents & Risks, Next Steps (1–3 bullets)."
        )
        user = [
            {"type": "text", "text": "CLIENT GUIDELINES:\n" + (rules_text if rules_present else "")[:9000]},
            {"type": "text", "text": "\n\nESTIMATE TEXT:\n" + (est_text or "")[:12000]},
        ]
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        rsp = openai_chat(messages, max_tokens=700)

    elif intent == "comprehensive":
        system = (
            "Comprehensive audit. Compare client guidelines ↔ estimate AND estimate ↔ photos. "
            "Do not restate guidelines; list where the estimate COMPLIES vs DEVIATES with short quotes from estimate text. "
            "If no rules provided, say 'No client rules provided' and skip compliance scoring vs rules. "
            "If no photos, omit all photo sections. "
            "VIN Verification line must be exactly one of: MATCH / MISMATCH / NOT VERIFIED / PHOTOS NOT PROVIDED. "
            "Sections: Estimate Snapshot; VIN Verification: <...>; Compliance vs Guidelines (if rules present); "
            "Estimate ↔ Photos (damage match with LEFT/RIGHT & panels, discrepancies, missing angles/measurements); "
            "Missing Documents & Risks; Next Steps (1–3 bullets). Always end with: Final Evaluation: NN%."
        )
        user = [
            {"type": "text", "text": "CLIENT GUIDELINES:\n" + (rules_text if rules_present else "")[:8000]},
            {"type": "text", "text": "\n\nESTIMATE TEXT:\n" + (est_text or "")[:12000]},
        ]
        if photos_present:
            user += img_payload()
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        rsp = openai_chat(messages, max_tokens=1100)

    elif intent == "photos_only":
        if not photos_present:
            rsp = None
            gpt_output = "No photos were provided with this request.\n\nFinal Evaluation: 0%"
        else:
            system = (
                "Compare photos to the estimate only. Identify primary damage location(s) using LEFT/RIGHT & panel names. "
                "Sections: Photo Coverage; Visible Damage vs Estimate; Discrepancies; Summary. End with: Final Evaluation: NN%."
            )
            user = [{"type": "text", "text": "ESTIMATE TEXT:\n" + (est_text or "")[:9000]}] + img_payload()
            messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            rsp = openai_chat(messages, max_tokens=800)

    elif intent == "invoices_with_photos":
        # Invoice guardrails: detect likely invoices/supplements by filename/header; never treat estimate as invoice
        inv_docs = collect_invoices(pdfs, est_pdf[0] if est_pdf else None)
        inv_text = ""
        inv_names = []
        for nm, blob, hit in inv_docs:
            inv_names.append(nm)
            inv_text += f"[{nm} via {hit}]\n" + first_page_ocr(blob) + "\n" + fast_pdf_text(blob, limit_pages=2) + "\n\n"

        system = (
            "Analyze supplements/invoices against the estimate (and photos if provided). "
            "First, list the invoice/supplement sources you actually detected (by filename) with 1–2 quoted header lines per file. "
            "If none were detected, write 'Invoices: Not provided' and DO NOT claim invoice content exists. "
            "An estimate is NOT an invoice—NEVER say 'invoice lists...' unless the invoice text includes those details. "
            "Only use invoice text given here; do not infer totals or line items. "
            "Sections: Invoices Detected; Invoices Summary; Support vs Estimate Lines (cite estimate lines by short quote); "
            "Photo Corroboration (if photos); Missing Documentation; Summary. End with: Final Evaluation: NN%."
        )

        user = [
            {"type": "text", "text": "ESTIMATE TEXT:\n" + (est_text or "")[:8000]},
            {"type": "text", "text": "\n\nINVOICES DETECTED (filenames):\n" + ("\n".join(inv_names) if inv_names else "None")},
            {"type": "text", "text": "\n\nINVOICES TEXT (use ONLY this content):\n" + (inv_text or "No invoices provided.")[:9000]},
        ]
        if photos_present:
            user += img_payload()

        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        rsp = openai_chat(messages, max_tokens=900)

    else:  # docs_checklist
        system = (
            "Documentation checklist only based on estimate text (and photos if provided). "
            "If no rules provided, say 'No client rules provided' and check only general documentation. "
            "Mark Present / Missing / Not found. End with: Final Evaluation: NN%."
        )
        user = [
            {"type": "text", "text": "CLIENT GUIDELINES:\n" + (rules_text if rules_present else "")[:6000]},
            {"type": "text", "text": "\n\nESTIMATE TEXT:\n" + (est_text or "")[:8000]},
        ]
        if photos_present:
            user += img_payload(6)
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        rsp = openai_chat(messages, max_tokens=600)

    gpt_output = (rsp.choices[0].message.content if rsp else locals().get("gpt_output", "Automated narrative unavailable.")).strip()

    # VIN verification line: honor GPT if present; otherwise derive from photos presence
    vin_ver = parse_vin_verification(gpt_output)
    if not vin_ver:
        vin_ver = ("PHOTOS NOT PROVIDED" if not photos_present else "NOT VERIFIED")

    comp = final_percent(gpt_output)

    # ======================= PDF =======================
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, sanitize_latin1("NSPXN.com AI Review Report"), ln=True, align="C")
    pdf.ln(5)
    pdf.set_font_size(10)
    pdf.multi_cell(0, 6, sanitize_latin1(f"File Number: {file_number}"))
    pdf.multi_cell(0, 6, sanitize_latin1(f"IA Company: {ia_company}"))
    pdf.multi_cell(0, 6, sanitize_latin1(f"Request Type: {request_type_label}"))
    pdf.multi_cell(0, 6, sanitize_latin1(f"Appraiser ID #: {appraiser_id}"))
    pdf.ln(4)
    pdf.multi_cell(0, 6, sanitize_latin1(f"Claim #: {claim or 'N/A'}"))
    pdf.multi_cell(0, 6, sanitize_latin1(f"VIN (from estimate): {vin}"))
    pdf.multi_cell(0, 6, sanitize_latin1(f"VIN verification (estimate vs photo): {vin_ver}"))
    pdf.multi_cell(0, 6, sanitize_latin1(f"Vehicle: {vehicle}"))
    if mileage:
        pdf.multi_cell(0, 6, sanitize_latin1(f"Odometer (from estimate): {mileage}"))
    if days is not None:
        pdf.multi_cell(0, 6, sanitize_latin1(f"Days to Repair (reported): {days}"))
    pdf.multi_cell(0, 6, sanitize_latin1(f"Compliance Score: {comp if comp is not None else 'N/A'}%"))
    if est_total:
        pdf.multi_cell(0, 6, sanitize_latin1(f"Estimate Printed Total: ${est_total}"))
    pdf.ln(4)
    pdf.set_font_size(12)
    pdf.cell(0, 8, sanitize_latin1("AI-4-IA Review Summary"), ln=True)
    pdf.set_font_size(10)
    pdf.multi_cell(0, 6, sanitize_latin1(gpt_output or "No narrative generated."))
    pdf.ln(4)
    pdf.set_font_size(12)
    pdf.cell(0, 8, sanitize_latin1("Estimate <-> Photos Consistency Review"), ln=True)
    pdf.set_font_size(10)
    pdf.multi_cell(
        0,
        6,
        sanitize_latin1("Included in narrative above.")
        if (intent in ("comprehensive", "photos_only", "invoices_with_photos") and photos_present)
        else sanitize_latin1("Not requested or no photos provided."),
    )

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_url = f"/download-pdf?file_number={file_number}"

    # ======================= Email (same structure) =======================
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim or 'N/A'}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Request Type: {request_type_label}

Claim #: {claim or 'N/A'}
VIN (from estimate): {vin}
VIN verification (estimate vs photo): {vin_ver}
Vehicle: {vehicle}
{('Odometer (from estimate): ' + str(mileage)) if mileage else ''}
{('Days to Repair (reported): ' + str(days)) if days is not None else ''}

Compliance Score: {(str(comp)+'%') if comp is not None else 'N/A'}
{('Estimate Printed Total: $' + str(est_total)) if est_total else ''}

Summary:
{gpt_output}
"""
        msg.set_content(body)
        smtp_pass = os.getenv("NSPXN_SMTP_PASS", "grr2025GRR")
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", smtp_pass)
            smtp.send_message(msg)
    except Exception:
        pass

    return {
        "request_type": request_type_label,
        "pdf_url": pdf_url,
        "pdf_filename": f"{file_number}.pdf",
        "pdf_b64": pdf_b64,
        "gpt_output": gpt_output,
        "file_number": file_number,
        "claim_number": claim or "N/A",
        "vehicle": vehicle,
        "vin_estimate": vin,
        "vin_verification": vin_ver,
        "odometer_estimate": mileage or "Not documented",
        "days_to_repair": days or "Not documented",
        "compliance_score": comp if comp is not None else "N/A",
        "valuation_present": valuation_present,
        "valuation_sources": valuation_sources,
        "rules_present": rules_present,
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    safe = re.sub(r"[^\w.\-]+", "-", file_number).strip("-_.")
    path = os.path.join(PDF_DIR, f"{safe}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{safe}.pdf")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

# --- Compatibility shims for legacy callsites ---
def extract_vin_from_text(text: str):
    try:
        return vin_from_text(text)
    except Exception:
        return None

def extract_vehicle_line_from_first_page(text: str):
    try:
        return vehicle_from_text(text)
    except Exception:
        return None
