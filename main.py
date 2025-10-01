from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
import os, io, re, time, base64, zipfile, smtplib, json, logging

from email.message import EmailMessage
from fpdf import FPDF
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat

from openai import OpenAI

# ======================= Config & Logging =======================
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)

OPENAI_MODEL = os.getenv("OAI_MODEL", "gpt-4o-mini")
OPENAI_FALLBACK = "gpt-3.5-turbo"
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))

SMTP_HOST = os.getenv("SMTP_HOST", "mail.tierra.net")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "info@nspxn.com")
SMTP_PASS = os.getenv("SMTP_PASS", "grr2025GRR")  # default to preserve prior behavior
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)     # default From = authenticated user
SMTP_TO = os.getenv("SMTP_TO", "info@nspxn.com")  # comma-separated
SEND_EMAIL = os.getenv("SEND_EMAIL", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai4ia")

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT)

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

# ======================= Utilities =======================
def _pp(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.85)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    return img

def sanitize_latin1(s: str) -> str:
    if not s: return ""
    repl = {
        "\u2018":"'","\u2019":"'","\u201C":'"',"\u201D":'"',
        "\u2013":"-","\u2014":"-","\u2022":"-","\u2026":"...",
        "\u2192":"->","\u2194":"<->","\u00A0":" "
    }
    for k,v in repl.items():
        s = s.replace(k,v)
    return s.encode("latin-1","ignore").decode("latin-1","ignore")

def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.\-]+","-", s)
    return s.strip("-_.") or f"report-{int(time.time())}"

def fast_pdf_text(pdf_bytes: bytes, limit_pages: int = 12) -> str:
    out = []
    try:
        rdr = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for i, pg in enumerate(rdr.pages[:limit_pages], 1):
            t = pg.extract_text() or ""
            if t.strip():
                out.append(f"[Page {i}]\n{t}")
    except Exception as e:
        log.info(f"PyPDF2 read fallback: {e}")
    return "\n\n".join(out)

def ocr_first_pages(pdf_bytes: bytes, max_pages: int = 3, dpi: int = 250) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
    except Exception as e:
        log.info(f"pdf2image fail: {e}"); return ""
    out = []
    for i, pg in enumerate(pages, 1):
        try:
            out.append(f"[OCR {i}]\n" + pytesseract.image_to_string(_pp(pg), lang="eng", config="--psm 6"))
        except Exception as e:
            log.info(f"OCR page {i} fail: {e}")
    return "\n".join(out)

def ocr_pages_for_vin(pdf_bytes: bytes, max_pages: int = 4, dpi: int = 250) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
    except Exception: return ""
    out = []
    for im in pages:
        try:
            out.append(pytesseract.image_to_string(_pp(im), lang="eng", config="--psm 6"))
        except Exception: pass
    return "\n".join(out)

# ---------- VIN / Claim / Vehicle ----------
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
VIN_TIGHT = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
VIN_RELAX = re.compile(
    r"(?:V\.?I\.?N\.?|Vehicle\s+Identification\s+Number|VIN)\b[^A-Z0-9]{0,40}((?:[A-HJ-NPR-Z0-9][\s\-]*){17})",
    re.IGNORECASE,
)
_trans = {**{str(i): i for i in range(10)},
          **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_w = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def _norm_vin(s: str) -> Optional[str]:
    s = (s or "").upper()
    s = re.sub(r"[^A-HJ-NPR-Z0-9]","", s).replace("O","0").replace("I","1").replace("Q","0")
    if len(s)!=17 or any(ch not in VIN_ALLOWED for ch in s): return None
    return s

def _vin_ok(v: str) -> bool:
    try:
        tot = sum(_trans[ch]*_w[i] for i,ch in enumerate(v))
        chk = tot % 11
        return v[8] == ("X" if chk==10 else str(chk))
    except Exception:
        return False

def vin_from_text(text: str) -> Optional[str]:
    cands = [m.group(1) for m in VIN_RELAX.finditer(text or "")] + VIN_TIGHT.findall(text or "")
    uniq, seen = [], set()
    for c in cands:
        v = _norm_vin(c)
        if v and v not in seen: uniq.append(v); seen.add(v)
    for v in uniq:
        if _vin_ok(v): return v
    return uniq[0] if uniq else None

def scan_estimate_for_vin(pdf_bytes: bytes) -> Optional[str]:
    text = fast_pdf_text(pdf_bytes, limit_pages=12)
    v = vin_from_text(text)
    if v: return v
    ocr = ocr_pages_for_vin(pdf_bytes, max_pages=4, dpi=250)
    v = vin_from_text(text + "\n" + ocr)
    if v: return v
    for c in re.findall(r"\b([A-HJ-NPR-Z0-9\-\s]{17,40})\b", text+"\n"+ocr, flags=re.I):
        cc = re.sub(r"[^A-HJ-NPR-Z0-9]","", c.upper()).replace("O","0").replace("I","1").replace("Q","0")
        if len(cc)==17 and re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", cc): return cc
    return None

def claim_from_text_strict(text: str) -> Optional[str]:
    """Direct 'Claim #:' grab across the whole text (handles line-wrap)."""
    if not text: return None
    # Same-line
    m = re.search(r"(?is)Claim\s*#\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-/\s]*\d[A-Za-z0-9\-/\s]*)", text)
    if m:
        cand = m.group(1).strip()
        cand = re.sub(r"\s+", "", cand)
        if len(re.findall(r"\d", cand))>=2 and 5<=len(cand)<=40:
            return m.group(1).strip().rstrip(".:,;")
    # Next-line
    m2 = re.search(r"(?is)(Claim\s*#\s*[:\-]?)\s*\n\s*([A-Za-z0-9][A-Za-z0-9\-/\s]*\d[A-Za-z0-9\-/\s]*)", text)
    if m2:
        cand = m2.group(2).strip()
        cand = re.sub(r"\s+", "", cand)
        if len(re.findall(r"\d", cand))>=2 and 5<=len(cand)<=40:
            return m2.group(2).strip().rstrip(".:,;")
    return None

def claim_from_text_smart(text: str) -> Optional[str]:
    if not text: return None
    hit = claim_from_text_strict(text)
    if hit: return hit
    lines = [re.sub(r"\s+"," ", ln.strip()) for ln in (text or "").splitlines()]
    for i, ln in enumerate(lines):
        if re.search(r"\b(Claim|Assignment|Reference|File|RO|Work\s*Order|Loss)\b", ln, re.I):
            m = re.search(r"(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/\s]*\d[A-Za-z0-9\-_\/\s]*)", ln, re.I)
            cand = (m.group(1).strip() if m else "")
            if not cand or len(re.findall(r"\d", cand))<2:
                if i+1 < len(lines):
                    nxt = lines[i+1]
                    m2 = re.search(r"([A-Za-z0-9][A-Za-z0-9\-_\/\s]*\d[A-Za-z0-9\-_\/\s]*)", nxt)
                    cand = (m2.group(1).strip() if m2 else "")
            if cand:
                cleaned = re.sub(r"\s+","", cand).strip(".:,;#")
                if len(re.findall(r"\d", cleaned))>=2 and 5<=len(cleaned)<=40:
                    return cand.strip().rstrip(".:,;")
    flat = re.sub(r"[\r\n]+"," ", text)
    for m in re.finditer(r"\b\d{4,}-\d{5,}\b", flat):
        s,e = m.start(), m.end()
        window = flat[max(0,s-120):min(len(flat), e+120)]
        if re.search(r"\b(Claim|Assignment|Reference|File|RO|Work\s*Order|Loss)\b", window, re.I):
            return m.group(0)
    return None

MAKE_MAP = {"NISSAN":"Nissan","CHEV":"Chevrolet","CHEVY":"Chevrolet","TOYOTA":"Toyota","FORD":"Ford","HONDA":"Honda",
            "HYUNDAI":"Hyundai","KIA":"Kia","BMW":"BMW","MERCEDES":"Mercedes-Benz","MB":"Mercedes-Benz",
            "VW":"Volkswagen","VOLKS":"Volkswagen","SUBARU":"Subaru","MAZDA":"Mazda","DODGE":"Dodge","RAM":"Ram","JEEP":"Jeep"}
STOP = {"GASOLINE","DIESEL","HYBRID","ELECTRIC","BLACK","WHITE","BLUE","RED","SILVER","GRAY","GREY",
        "4D","2D","SED","SDN","SUV","COUPE","HATCH","TRUCK","WAGON","AWD","FWD","RWD","L","GDI","TURBO","PAINT","CLEAR","COAT","COLOR"}

def vehicle_from_text(text: str) -> Optional[str]:
    for ln in (text or "").splitlines():
        ln = re.sub(r"\s{2,}"," ", ln.strip())
        if re.match(r"^\s*(19|20)\d{2}\b", ln) and not re.search(r"\b(AM|PM)\b", ln):
            parts = ln.split()
            year = parts[0]; tail = parts[1:]
            keep = []
            for t in tail:
                raw = re.sub(r"[^\w\-]","", t).upper()
                if raw in STOP or raw in ("A/M","OEM"): break
                keep.append(t)
                if len(keep)>=4: break
            if keep:
                mk = MAKE_MAP.get(keep[0].upper(), keep[0].capitalize())
                return " ".join([year, mk] + keep[1:])
    return None

def vin_from_photos(images: List[Tuple[str,bytes]]) -> Optional[str]:
    cands = []
    for _, blob in images:
        try:
            im = Image.open(io.BytesIO(blob))
            for r in (0,90,180,270):
                img = im.rotate(r, expand=True)
                txt = pytesseract.image_to_string(_pp(img), lang="eng", config="--psm 7")
                cands.extend(re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", txt))
        except Exception: pass
    for c in cands:
        v = _norm_vin(c)
        if v and _vin_ok(v): return v
    for c in cands:
        v = _norm_vin(c)
        if v: return v
    return None

def harvest_photos_from_pdf(pdf_bytes: bytes, need_photos: bool, max_pages: int = 10) -> List[Tuple[str,bytes]]:
    if not need_photos: return []
    out: List[Tuple[str,bytes]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)[:max_pages]
        for i, page in enumerate(pages, 1):
            g = _pp(page)
            var = ImageStat.Stat(g).var[0] if g.mode=="L" else sum(ImageStat.Stat(g).var)/3
            looks_photo = var > 120
            if looks_photo:
                buf = io.BytesIO(); page.save(buf, format="JPEG", quality=85)
                out.append((f"pdf-photo-p{i}.jpg", buf.getvalue()))
    except Exception as e:
        log.info(f"harvest photos error: {e}")
    return out

def detect_valuations(pdfs: List[Tuple[str, bytes]], est_pdf_name: Optional[str]) -> Tuple[bool, list, str]:
    """Light check for KBB/NADA/Black Book/Edmunds; return (present?, source list, text snippet)."""
    VAL_KEYS = [("Kelley Blue Book","KBB"),("KBB.com","KBB"),("NADA","NADA/J.D. Power"),("J.D. Power","NADA/J.D. Power"),("Black Book","Black Book"),("Edmunds","Edmunds")]
    GENERIC_PAT = re.compile(r"(Average Price|Estimated Trade-?In Value|Clean Retail Value|Vehicle Valuation)", re.I)
    sources = []; snippet = ""
    for nm, blob in pdfs:
        if est_pdf_name and nm == est_pdf_name: continue
        try:
            txt = (ocr_first_pages(blob, max_pages=1) + "\n" + fast_pdf_text(blob, limit_pages=1))
        except Exception:
            txt = ""
        up = txt.upper()
        hit = None
        for key, lab in VAL_KEYS:
            if key.upper() in up:
                hit = lab; break
        if hit is None and GENERIC_PAT.search(up):
            hit = "Generic Valuation"
        if hit:
            sources.append(hit)
            if not snippet:
                snippet = txt[:600]
    sources = list(dict.fromkeys(sources))
    return (len(sources) > 0), sources, snippet

def final_percent(narr: str) -> Optional[int]:
    m = re.search(r"(final\s*(evaluation|score|compliance)\s*[:\-]?\s*)(\d{1,3})\s*%", narr or "", re.IGNORECASE)
    if m:
        try:
            v = int(m.group(3)); return v if 0<=v<=100 else None
        except Exception: return None
    return None

def openai_chat(messages, max_tokens=900):
    for attempt in range(2):
        try:
            return client.chat.completions.create(model=OPENAI_MODEL, messages=messages, max_tokens=max_tokens, temperature=0)
        except Exception as e:
            if "429" in str(e) or "timeout" in str(e).lower():
                time.sleep(1.0 * (attempt+1)); continue
            break
    try:
        return client.chat.completions.create(model=OPENAI_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=0)
    except Exception as e:
        log.warning(f"fallback model failed: {e}")
        return None

# ======================= API =======================
@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("guidelines_only")
):
    intent = ai_intent if ai_intent in INTENTS else "guidelines_only"
    request_type_label = INTENTS[intent]
    file_number = safe_filename(file_number)

    # --- Collect files ---
    pdfs: List[Tuple[str,bytes]] = []
    images: List[Tuple[str,bytes]] = []
    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            pdfs.append((name, raw))
        elif name.endswith((".jpg",".jpeg",".png",".webp",".tif",".tiff")):
            images.append((name, raw))
        elif name.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for zi in z.infolist():
                        if zi.is_dir(): continue
                        zname = zi.filename.lower()
                        zdata = z.read(zi)
                        if zname.endswith(".pdf"):
                            pdfs.append((zname, zdata))
                        elif zname.endswith((".jpg",".jpeg",".png",".webp",".tif",".tiff")):
                            images.append((zname, zdata))
            except Exception as e:
                log.info(f"zip read error: {e}")

    # Identify estimate
    est_pdf = None
    for nm, blob in pdfs:
        if "est" in nm or "estimate" in nm:
            est_pdf = (nm, blob); break
    if est_pdf is None and pdfs: est_pdf = pdfs[0]

    need_photos = intent in ("comprehensive","photos_only","invoices_with_photos")
    if est_pdf:
        images += harvest_photos_from_pdf(est_pdf[1], need_photos=need_photos, max_pages=8)

    photos_present = len(images) > 0

    # Text harvest
    est_text = ""
    if est_pdf:
        est_text = fast_pdf_text(est_pdf[1], limit_pages=12)
        est_text += "\n\n" + ocr_first_pages(est_pdf[1], max_pages=3, dpi=250)

    all_text = est_text
    for nm, blob in pdfs:
        if not est_pdf or nm != est_pdf[0]:
            all_text += "\n\n" + ocr_first_pages(blob, max_pages=1, dpi=250)

    vin_estimate = scan_estimate_for_vin(est_pdf[1]) if est_pdf else None
    vin = vin_estimate or vin_from_text(est_text) or vin_from_text(all_text) or "N/A"
    claim = claim_from_text_smart(est_text) or claim_from_text_smart(all_text) or "N/A"
    vehicle = vehicle_from_text(est_text) or vehicle_from_text(all_text) or "N/A"

    vin_photo = vin_from_photos(images) if photos_present else None
    if photos_present:
        if vin_estimate and vin_photo:
            vin_ver = "MATCH" if _norm_vin(vin_estimate)==_norm_vin(vin_photo) else "MISMATCH"
        else:
            vin_ver = "NOT VERIFIED"
    else:
        vin_ver = "PHOTOS NOT PROVIDED"

    rules_text = (client_rules or "").strip()
    rules_present = len(rules_text) > 0

    # Minimal valuation detection to ground GPT
    est_name = est_pdf[0] if est_pdf else None
    valuation_present, valuation_sources, valuation_snippet = detect_valuations(pdfs, est_name)

    # ======= GPT =======
    header_hint = {
        "intent": intent,
        "photos_present": photos_present,
        "valuation_present": valuation_present,
        "valuation_sources": valuation_sources,
        "vin_estimate": vin if vin!="N/A" else "",
        "claim": claim if claim!="N/A" else "",
        "vehicle": vehicle,
        "rules_present": rules_present,
    }

    sys_common = (
        "You are an auto-claims appraisal assistant. Use ONLY the provided estimate text, uploaded photos, and pasted client rules.\n"
        "- If photos_present==false: you MUST write 'No photos were provided.' in any photo section and avoid claiming any photo types were present.\n"
        "- If valuation_present==false: do NOT claim NADA/KBB/valuation was included.\n"
        "- Never assert a document/photo exists unless you can quote it from the provided texts; include a short 3–12 word quote for every 'present' claim.\n"
        "- Do NOT discuss salvage bids unless the estimate explicitly declares total loss.\n"
        "- For supplements (invoices_with_photos) do NOT require VIN/registration/odometer photos.\n"
        "Always end with exactly one line: Final Evaluation: NN%."
    )

    messages = [{"role":"system","content": sys_common + "\nParsed header: " + json.dumps(header_hint)}]

    def img_payload(max_imgs=12):
        out = []
        for _, blob in images[:max_imgs]:
            b64 = base64.b64encode(blob).decode("utf-8")
            out.append({"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}})
        return out

    if intent == "guidelines_only":
        sys2 = (
            "Compare client guidelines to the ESTIMATE only.\n"
            "Sections:\n"
            "• Estimate Snapshot (year/make/model if quoted, total if quoted)\n"
            "• Compliance vs Guidelines (Compliant/Non-compliant/Not found WITH short quote from ESTIMATE TEXT)\n"
            "• Missing Documents & Risks\n"
            "• Next Steps (1–3 bullets)"
        )
        user = [
            {"type":"text","text": "CLIENT GUIDELINES:\n" + (rules_text if rules_present else "")[:9000]},
            {"type":"text","text": "\n\nESTIMATE TEXT:\n" + (est_text or "")[:12000]},
            {"type":"text","text": "\n\nVALUATION TEXT SNIPPET (if any):\n" + (valuation_snippet if valuation_present else "")}
        ]
        messages.append({"role":"system","content": sys2})
        messages.append({"role":"user","content": user})
        rsp = openai_chat(messages, max_tokens=900)

    elif intent == "comprehensive":
        sys2 = (
            "Comprehensive audit: guidelines ↔ estimate AND estimate ↔ photos.\n"
            "VIN Verification must be exactly one of: MATCH / MISMATCH / NOT VERIFIED / PHOTOS NOT PROVIDED.\n"
            "Sections:\n"
            "• Estimate Snapshot (quote VIN/vehicle/mileage/total only if visible in ESTIMATE TEXT; include short quotes)\n"
            "• VIN Verification: <...>\n"
            "• Compliance vs Guidelines (if rules_present; each 'present' must include a short quote)\n"
            "• Estimate ↔ Photos (damage area; LEFT/RIGHT + key panels; discrepancies; missing angles/measurements)\n"
            "• Missing Documents & Risks\n"
            "• Next Steps (1–3 bullets)"
        )
        user = [
            {"type":"text","text": "CLIENT GUIDELINES:\n" + (rules_text if rules_present else "")[:8000]},
            {"type":"text","text": "\n\nESTIMATE TEXT:\n" + (est_text or "")[:12000]},
            {"type":"text","text": "\n\nVALUATION TEXT SNIPPET (if any):\n" + (valuation_snippet if valuation_present else "")}
        ] + (img_payload() if photos_present else [])
        messages.append({"role":"system","content": sys2})
        messages.append({"role":"user","content": user})
        rsp = openai_chat(messages, max_tokens=1200)

    elif intent == "photos_only":
        if not photos_present:
            rsp = None
            gpt_output = "No photos were provided.\n\nFinal Evaluation: 0%"
        else:
            sys2 = ("Compare photos to estimate. Sections: Photo Coverage; Visible Damage vs Estimate; Discrepancies; Summary.")
            user = [{"type":"text","text": "ESTIMATE TEXT:\n" + (est_text or "")[:9000]}] + img_payload()
            messages.append({"role":"system","content": sys2})
            messages.append({"role":"user","content": user})
            rsp = openai_chat(messages, max_tokens=900)

    elif intent == "invoices_with_photos":
        sys2 = ("Supplement/invoices vs estimate (and photos if provided). Only use text provided; never infer. If no invoices detected, say so plainly.")
        user = [{"type":"text","text": "ESTIMATE TEXT:\n" + (est_text or "")[:9000]}] + (img_payload() if photos_present else [])
        messages.append({"role":"system","content": sys2})
        messages.append({"role":"user","content": user})
        rsp = openai_chat(messages, max_tokens=900)

    else:
        sys2 = ("Documentation checklist based on available materials. Mark Present/Missing/Not found with short quotes when present.")
        user = [
            {"type":"text","text": "CLIENT GUIDELINES:\n" + (rules_text if rules_present else "")[:6000]},
            {"type":"text","text": "\n\nESTIMATE TEXT:\n" + (est_text or "")[:8000]}
        ] + (img_payload(6) if photos_present else [])
        messages.append({"role":"system","content": sys2})
        messages.append({"role":"user","content": user})
        rsp = openai_chat(messages, max_tokens=800)

    gpt_output = (rsp.choices[0].message.content if rsp else locals().get("gpt_output","")).strip() or "Automated narrative unavailable."
    comp = final_percent(gpt_output)

    # ======================= PDF =======================
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, sanitize_latin1("NSPXN.com AI Review Report"), ln=True, align="C")
    pdf.ln(4); pdf.set_font_size(10)
    pdf.multi_cell(0,6, sanitize_latin1(f"File Number: {file_number}"))
    pdf.multi_cell(0,6, sanitize_latin1(f"IA Company: {ia_company}"))
    pdf.multi_cell(0,6, sanitize_latin1(f"Request Type: {request_type_label}"))
    pdf.multi_cell(0,6, sanitize_latin1(f"Appraiser ID #: {appraiser_id}"))
    pdf.ln(2)
    pdf.multi_cell(0,6, sanitize_latin1(f"Claim #: {claim or 'N/A'}"))
    pdf.multi_cell(0,6, sanitize_latin1(f"VIN (from estimate): {vin}"))
    pdf.multi_cell(0,6, sanitize_latin1(f"VIN verification (estimate vs photo): {vin_ver}"))
    pdf.multi_cell(0,6, sanitize_latin1(f"Vehicle: {vehicle}"))
    pdf.multi_cell(0,6, sanitize_latin1(f"Compliance Score: {comp if comp is not None else 'N/A'}%"))
    pdf.ln(3); pdf.set_font_size(12); pdf.cell(0,8, sanitize_latin1("AI-4-IA Review Summary"), ln=True)
    pdf.set_font_size(10); pdf.multi_cell(0,6, sanitize_latin1(gpt_output))

    pdf_bytes = pdf.output(dest="S").encode("latin-1","ignore")
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    with open(pdf_path, "wb") as f: f.write(pdf_bytes)
    pdf_url = f"/download-pdf?file_number={file_number}"
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # ======================= Email =======================
    try:
        if SEND_EMAIL == "1":
            msg = EmailMessage()
            msg["Subject"] = f"AI-4-IA Review: {claim or 'N/A'}"
            msg["From"] = SMTP_FROM
            msg["To"] = SMTP_TO
            msg["Reply-To"] = SMTP_USER
            body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Request Type: {request_type_label}

Claim #: {claim or 'N/A'}
VIN (from estimate): {vin}
VIN verification (estimate vs photo): {vin_ver}
Vehicle: {vehicle}

Compliance Score: {(str(comp)+'%') if comp is not None else 'N/A'}

Summary:
{gpt_output}
"""
            msg.set_content(body)
            # attach PDF
            msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=f"{file_number}.pdf")
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.login(SMTP_USER, SMTP_PASS)
                tos = [t.strip() for t in SMTP_TO.split(",") if t.strip()]
                if tos:
                    msg["To"] = ", ".join(tos)
                smtp.send_message(msg)
    except Exception as e:
        log.info(f"email send failed: {e}")

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
        "compliance_score": comp if comp is not None else "N/A",
        "photos_present": photos_present
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    safe = re.sub(r"[^\w.\-]+","-", file_number).strip("-_.")
    path = os.path.join(PDF_DIR, f"{safe}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{safe}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})
