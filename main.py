from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os, io, re, json, base64, math, datetime, hashlib, logging, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat

# ---------- OPTIONAL fast embedded text ----------
try:
    from PyPDF2 import PdfReader
    HAVE_PYPDF2 = True
except Exception:
    HAVE_PYPDF2 = False

# ---------- OpenAI ----------
from openai import OpenAI
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY env var missing")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o"

# ---------- App ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com","https://www.nspxn.com","http://nspxn.com","http://www.nspxn.com",
        "https://nspxn.onrender.com"
    ],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="app.log", filemode="a"
)
log = logging.getLogger("ai4ia")

# -------------------- Helpers: OCR + images --------------------
def _pre(img: Image.Image) -> Image.Image:
    g = img.convert("L")
    g = ImageEnhance.Contrast(g).enhance(2.0)
    g = ImageOps.autocontrast(g.filter(ImageFilter.MedianFilter(3)))
    return g

def ocr_image(img: Image.Image, psm=6) -> str:
    try:
        return pytesseract.image_to_string(_pre(img), lang="eng", config=f"--psm {psm} --oem 1")
    except Exception:
        return ""

def pdf_text_fast(pdf_bytes: bytes, max_pages_ocr: int = 3) -> str:
    """Try embedded text first (fast). If empty, light OCR for 1–3 pages."""
    text = ""
    if HAVE_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            parts = []
            for i, p in enumerate(reader.pages):
                parts.append(p.extract_text() or "")
                if i == 0:  # we mainly need the first page
                    break
            text = "\n".join(parts).strip()
        except Exception as e:
            log.warning(f"PyPDF2 failed: {e}")
    if text:
        return text
    # fallback, very light OCR (first up to 3 pages)
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=150)
        out = []
        for i, im in enumerate(pages[:max_pages_ocr], start=1):
            t = ocr_image(im, psm=6)
            if not t.strip():
                t = ocr_image(im, psm=3)
            if t.strip():
                out.append(t)
        return "\n".join(out)
    except Exception as e:
        log.error(f"OCR fallback failed: {e}")
        return ""

def harvest_photos_from_pdf(pdf_bytes: bytes, dpi: int = 135, max_pages: int = 20) -> List[Tuple[str, bytes]]:
    """Extract photo-like pages from a non-estimate PDF (for vision)."""
    out = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
        for i, im in enumerate(pages, 1):
            g = im.convert("L")
            var = ImageStat.Stat(g).var[0]
            if var > 110:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
                out.append((f"pdf-photo-p{i}.jpg", buf.getvalue()))
    except Exception as e:
        log.warning(f"harvest_photos_from_pdf: {e}")
    return out

# -------------------- VIN utilities --------------------
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
def normalize_vin(s: str) -> Optional[str]:
    s = re.sub(r"[^A-HJ-NPR-Z0-9]", "", (s or "").upper())
    s = s.replace("O","0").replace("I","1").replace("Q","0")
    return s if len(s) == 17 and all(c in VIN_ALLOWED for c in s) else None

_trans = {**{str(i): i for i in range(10)},
          **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,
                 S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def vin_checksum_ok(v: str) -> bool:
    try:
        s = 0
        for i, ch in enumerate(v):
            s += _trans[ch] * _weights[i]
        chk = s % 11
        return v[8] == ("X" if chk == 10 else str(chk))
    except Exception:
        return False

def best_vin(cands: List[str]) -> Optional[str]:
    for c in cands:
        v = normalize_vin(c)
        if v and vin_checksum_ok(v):
            return v
    return None

VIN_WORD = re.compile(r'(?i)\bV[\W_]*I[\W_]*N\b')
VIN_17 = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b')
VIN_SEQ = re.compile(r'(?i)((?:[A-HJ-NPR-Z0-9][\s\.\-–—:_]){16}[A-HJ-NPR-Z0-9])')

def extract_vin_from_text_firstpage(text: str) -> Optional[str]:
    if not text: return None
    # Prefer a window near "VIN"
    pos = [m.end() for m in VIN_WORD.finditer(text)]
    for p in pos:
        win = text[p:p+220]
        for m in VIN_SEQ.finditer(win):
            v = normalize_vin(m.group(1))
            if v and vin_checksum_ok(v): return v
        v = best_vin(VIN_17.findall(win))
        if v: return v
    # fallback anywhere
    for m in VIN_SEQ.finditer(text):
        v = normalize_vin(m.group(1))
        if v and vin_checksum_ok(v): return v
    return best_vin(VIN_17.findall(text))

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    # Check up to 8 likely VIN photos (heuristic via quick OCR sniff)
    likely = []
    for name, blob in image_blobs:
        try:
            im = Image.open(io.BytesIO(blob))
            im.thumbnail((1200, 1200))
            s = pytesseract.image_to_string(im, lang="eng", config="--psm 6 --oem 1")
            up = s.upper()
            if "VIN" in up or VIN_17.search(up) or VIN_SEQ.search(up):
                likely.append((name, blob))
        except Exception:
            pass
    likely = likely[:8]

    def variants(im: Image.Image) -> List[Image.Image]:
        g = im.convert("L")
        v = [
            ImageEnhance.Contrast(g).enhance(2.0),
            ImageEnhance.Sharpness(g).enhance(2.0),
            g.point(lambda p: 255 if p > 180 else 0, mode="1").convert("L"),
            ImageOps.autocontrast(g.filter(ImageFilter.MedianFilter(3))),
        ]
        return v

    for _, blob in likely:
        try:
            im = Image.open(io.BytesIO(blob))
            if im.width < 2000:
                h = int(im.height * (2000/im.width))
                im = im.resize((2000, h), Image.LANCZOS)
            text = []
            for v in variants(im):
                for psm in (7,6,11):
                    text.append(pytesseract.image_to_string(v, lang="eng", config=f"--psm {psm} --oem 1"))
            up = "\n".join(text).upper()
            for m in VIN_WORD.finditer(up):
                win = up[m.end():m.end()+220]
                for mm in VIN_SEQ.finditer(win):
                    v = normalize_vin(mm.group(1))
                    if v and vin_checksum_ok(v): return v
                v = best_vin(VIN_17.findall(win))
                if v: return v
            for mm in VIN_SEQ.finditer(up):
                v = normalize_vin(mm.group(1))
                if v and vin_checksum_ok(v): return v
            v = best_vin(VIN_17.findall(up))
            if v: return v
        except Exception:
            pass
    return None

# -------------------- Claim / Vehicle / Rates / Tax --------------------
CLAIM_RX = re.compile(r'(?is)\bclaim\b\W{0,5}(?:#:?|number)?\W{0,3}([A-Z0-9][A-Z0-9\-/\.]{2,60})')
CLAIM_BLACKLIST = {"SERVICE","SERVICES","PHONE","EMAIL","FAX","TOTAL","POLICY"}
def clean_claim(s: str) -> str:
    s = re.sub(r'[\s_]+', '', (s or '').strip(': ').strip())
    s = s.replace('–','-').replace('—','-')
    s = re.sub(r'(V\d+)$','',s,flags=re.I)
    return s
def valid_claim(s: str) -> bool:
    return len(s) >= 3 and re.search(r'\d', s) and s.upper() not in CLAIM_BLACKLIST

def extract_claim(text: str) -> Optional[str]:
    for m in CLAIM_RX.finditer(text or ""):
        c = clean_claim(m.group(1))
        if valid_claim(c): return c
    return None

def extract_vehicle_firstpage(text: str) -> Optional[str]:
    if not text: return None
    # Year Make Model, mileage
    m1 = re.search(r"\b(19|20)\d{2}\b.*?\b([A-Za-z]{3,})\b\s+([A-Za-z0-9\-]+)", text, re.I|re.S)
    mm = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text, re.I)
    if m1:
        year = re.search(r"(19|20)\d{2}", m1.group(0)).group(0)
        make = m1.group(2); model = m1.group(3)
        miles = mm.group(1) if mm else "unknown"
        return f"{year} {make} {model}, {miles} miles"
    return None

def labor_rates_present(text: str) -> bool:
    pats = [
        r"Body[^\n]{0,120}\$\s*\d{2,3}\.?(\d{2})?\s*(/hr|per\s*hour|hr)",
        r"Paint[^\n]{0,120}\$\s*\d{2,3}\.?(\d{2})?\s*(/hr|per\s*hour|hr)",
        r"Refinish|Supplies"
    ]
    return any(re.search(p, text or "", re.I) for p in pats)

def tax_present(text: str) -> bool:
    return re.search(r"(Sales\s*)?Tax[^\n]{0,60}(\d{1,2}\.\d{1,3}\s*%|\$\s*\d+(\.\d{2})?)", text or "", re.I) is not None

# -------------------- Client rules parse --------------------
def parse_client_rules(rules: str) -> Dict[str, Any]:
    u = (rules or "").lower()
    want_tax = bool(re.search(r"\btax\b.*(utilize|apply|required|must)", u))
    oem_only = bool(re.search(r"(oem[-\s]*only|no\s*aftermarket|no\s*non[-\s]*oem)", u))
    aftermarket_first = bool(re.search(r"(aftermarket.*(first|preferred)|use.*aftermarket.*before)", u))
    # required photos list
    req_photos = ["vin","odometer","license plate","four corners"]
    return {"require_tax": want_tax, "oem_only": oem_only, "aftermarket_first": aftermarket_first,
            "required_photos": req_photos}

# -------------------- Photo presence --------------------
PLATE_RX = re.compile(r'\b([A-Z0-9]{1,3}[-\s]?[A-Z0-9]{3,4}|[A-Z0-9]{5,8})\b')
def edges_exterior(img: Image.Image) -> bool:
    g = img.convert("L")
    var = ImageStat.Stat(g).var[0]
    evar = ImageStat.Stat(g.filter(ImageFilter.FIND_EDGES)).var[0]
    return var > 140 and evar > 400

def photo_presence(image_blobs: List[Tuple[str, bytes]], ocr_text: str) -> Dict[str, bool]:
    present = {"vin": False, "odometer": False, "license plate": False, "four corners": False}
    # VIN / ODO via photos OCR
    present["vin"] = extract_vin_from_photos(image_blobs) is not None
    # Odometer quick
    for _, b in image_blobs:
        try:
            im = Image.open(io.BytesIO(b)); im.thumbnail((1400, 1400))
            t = ocr_image(im, psm=7)
            if re.search(r"\b\d{2,6}\b.*(mi|miles|km)", t, re.I):
                present["odometer"] = True; break
        except Exception: pass
    # Plate
    for _, b in image_blobs[:24]:
        try:
            im = Image.open(io.BytesIO(b)); im.thumbnail((1600, 1600))
            t = pytesseract.image_to_string(_pre(im), lang="eng", config="--psm 7 --oem 1")
            if re.search(r"(license|registration)\s*plate", t, re.I) or PLATE_RX.search(t):
                present["license plate"] = True; break
        except Exception: pass
    # Four corners (edge richness heuristic)
    hits = 0
    for _, b in image_blobs[:40]:
        try:
            im = Image.open(io.BytesIO(b)); im.thumbnail((1600, 1600))
            if edges_exterior(im): hits += 1
        except Exception: pass
    if hits >= 3: present["four corners"] = True
    return present

# -------------------- Estimate items & vision compare (brief) --------------------
PANELS = ["bumper","fender","door","hood","grille","headlamp","headlight","taillamp","tail lamp",
          "quarter panel","rocker","roof","trunk","decklid","mirror","apron","radiator support",
          "wheel","tire","pillar","garnish","molding","fog lamp","reinforcement","cover","finish panel","combo lamp"]
OPS = ["replace","repair","refinish","r&i","r & i","align","blend","calibrate"]

def extract_items(text: str) -> List[Dict[str, str]]:
    out = []
    for line in (text or "").splitlines():
        l = line.lower().strip()
        if not l: continue
        if any(op in l for op in OPS) and any(p in l for p in PANELS):
            side = "unspecified"
            if " left" in l or re.search(r"\blh\b", l): side = "left"
            if " right" in l or re.search(r"\brh\b", l): side = "right"
            op = next((op for op in OPS if op in l), "operation")
            part = next((p for p in PANELS if p in l), "component")
            out.append({"op": op, "part": part, "side": side, "raw": line.strip()})
    # de-dup
    seen, uniq = set(), []
    for it in out:
        k = (it["op"], it["part"], it["side"])
        if k not in seen:
            uniq.append(it); seen.add(k)
    return uniq

def contact_sheets(image_blobs: List[Tuple[str, bytes]], cols=6, max_sheets=3, thumb_w=320) -> List[bytes]:
    if not image_blobs: return []
    thumbs = []
    for _, b in image_blobs:
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
            if im.width > thumb_w:
                h = int(im.height * (thumb_w/im.width))
                im = im.resize((thumb_w, h), Image.LANCZOS)
            thumbs.append(im)
        except Exception: pass
    if not thumbs: return []
    per_sheet = math.ceil(len(thumbs)/max_sheets)
    rows = math.ceil(per_sheet/cols)
    sheets = []
    i = 0
    for s in range(max_sheets):
        chunk = thumbs[i:i+per_sheet]
        if not chunk: break
        row_heights = []
        for r in range(rows):
            row = chunk[r*cols:(r+1)*cols]
            if not row: break
            row_heights.append(max(im.height for im in row))
        w = cols*thumb_w + (cols+1)*6
        h = sum(row_heights) + (len(row_heights)+1)*6
        canvas = Image.new("RGB", (w, h), (245,245,245))
        y = 6; pos = 0
        for rh in row_heights:
            x = 6
            for c in range(cols):
                if pos >= len(chunk): break
                im = chunk[pos]
                yoff = (rh - im.height)//2
                canvas.paste(im, (x, y+yoff))
                x += thumb_w + 6
                pos += 1
            y += rh + 6
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=68, optimize=True)
        sheets.append(buf.getvalue())
        i += per_sheet
    return sheets

def brief_photo_compare(items: List[Dict[str,str]], sheets: List[bytes]) -> Dict[str, Any]:
    if not items or not sheets:
        return {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "Comparison limited."}
    content: List[Dict[str, Any]] = [{"type":"text","text":"Estimate items:\n"+json.dumps(items, ensure_ascii=False)}]
    for b in sheets[:3]:
        content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b).decode("utf-8")}})
    system = (
        "You are an auto-damage visual auditor. "
        "Given brief estimate items and contact-sheet photos, mark for each item whether visible damage evidence exists. "
        "Return STRICT JSON with keys per_item[{op,part,side,photo_evidence,confidence,note}], "
        "not_in_photos[], extra_damage_in_photos[], overall."
    )
    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":content}],
            temperature=0, max_tokens=220
        )
        txt = (rsp.choices[0].message.content or "").strip()
        txt = txt.removeprefix("```json").removesuffix("```").strip()
        return json.loads(txt)
    except Exception as e:
        log.error(f"vision compare error: {e}")
        return {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "Comparison unavailable."}

# -------------------- Non-OEM detection --------------------
NON_OEM_RX = re.compile(r'\b(A/M|AFTER\s*MARKET|AFTERMARKET|LKQ|RECOND|RECONDITIONED|CAPA|ALT[-\s]*OE|REMAN)\b', re.I)
def non_oem_used(text: str) -> bool:
    for ln in (text or "").splitlines():
        L = ln.upper()
        if NON_OEM_RX.search(L) and any(p.upper() in L for p in [p.upper() for p in PANELS]):
            return True
    return False

# -------------------- PDF utils --------------------
def pdf_h1(pdf: FPDF, t: str):
    pdf.set_font_size(12); pdf.cell(0,8, t, ln=True); pdf.set_font_size(10)

def pdf_kv(pdf: FPDF, k: str, v: str):
    pdf.multi_cell(0,6, f"{k}: {v}")

# -------------------- API --------------------
@app.get("/")
def root():
    return PlainTextResponse("ok")

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...)
):
    # Collect
    first_pdf_bytes: Optional[bytes] = None
    first_pdf_text: str = ""
    all_texts: List[str] = []
    photos: List[Tuple[str, bytes]] = []

    for f in files:
        data = await f.read()
        name = (f.filename or "").lower()
        if name.endswith((".jpg",".jpeg",".png",".webp")):
            photos.append((name, data))
        elif name.endswith(".pdf"):
            if first_pdf_bytes is None:
                first_pdf_bytes = data
                first_pdf_text = pdf_text_fast(data)  # FIRST PAGE fast path
            # Only harvest photos if the PDF doesn't look like the estimate
            looks_est = bool(re.search(r'\bclaim\b', first_pdf_text, re.I) and re.search(r'\bvin\b', first_pdf_text, re.I))
            if not looks_est:
                photos += harvest_photos_from_pdf(data)
            # also keep full text (first page is enough for metadata/rates/tax)
            all_texts.append(first_pdf_text)
        elif name.endswith(".docx"):
            try:
                doc = Document(io.BytesIO(data))
                all_texts.append("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
            except Exception: pass
        elif name.endswith(".txt"):
            all_texts.append(data.decode("utf-8", errors="ignore"))

    est_text = first_pdf_text or "\n".join(all_texts)

    # --- Extract core fields from FIRST PAGE of estimate
    claim = extract_claim(first_pdf_text) or "N/A"
    vin_est = extract_vin_from_text_firstpage(first_pdf_text) or "N/A"
    vehicle = extract_vehicle_firstpage(first_pdf_text) or "N/A"

    # --- Labor & Tax
    has_rates = labor_rates_present(est_text)
    has_tax = tax_present(est_text)

    # --- VIN photo verification
    vin_photo = extract_vin_from_photos(photos)
    if vin_est != "N/A" and vin_photo:
        vin_verify = "Match" if vin_est == vin_photo else f"No Match (photo shows {vin_photo})"
    elif vin_est != "N/A" and not vin_photo:
        vin_verify = "VIN photo not found"
    elif vin_est == "N/A" and vin_photo:
        vin_verify = "VIN not found in estimate"
    else:
        vin_verify = "VIN unavailable"

    # --- Required photo presence (VIN / ODO / Plate / Four corners)
    req = parse_client_rules(client_rules)
    presence = photo_presence(photos, est_text)
    missing = [p for p in req["required_photos"] if not presence.get(p, False)]

    # --- Parts compliance from client rules
    parts_non_oem = non_oem_used(est_text)
    parts_deduction = 0
    parts_summary = ""
    if req["oem_only"]:
        if parts_non_oem:
            parts_deduction = -25
            parts_summary = "- Non-compliant: non-OEM parts on OEM-only vehicle per client rules."
        else:
            parts_summary = "- Compliant: OEM parts only per client rules."
    elif req["aftermarket_first"]:
        # If rules push aftermarket-first and we only see OEM lines (i.e., no non-OEM flags at all),
        # require justification string; if not present, mark as attention.
        if not parts_non_oem and not re.search(r"(alt|alternate|alternative|aftermarket not available|oem required|justif)", est_text, re.I):
            parts_summary = "- Attention: Rules prefer aftermarket first; estimate appears OEM-only—add justification for not using alternatives."
        else:
            parts_summary = "- Parts usage appears consistent with aftermarket-first preference."
    else:
        parts_summary = "- Parts appear OEM or not flagged as non-OEM."

    # --- Scoring
    score = 100
    if not has_rates: score -= 50
    if req["require_tax"] and not has_tax: score -= 25
    score += -25 * len(missing)
    score += parts_deduction
    score = max(0, min(100, score))

    # --- Brief estimate ↔ photos compare (contact-sheets + few tokens)
    items = extract_items(est_text)
    sheets = contact_sheets(photos, cols=6, max_sheets=3, thumb_w=320)
    compare = brief_photo_compare(items, sheets)

    # --- Build Summary (succinct, deterministic wording)
    photo_lines = ["- All required photo types present (four corners, VIN, odometer, plate)."] if not missing \
        else [f"- Missing: {', '.join(missing)}."]
    labor_lines = ["- Labor rates listed on estimate."] if has_rates else ["- Labor rates missing."]
    tax_lines = ["- Tax rate present on estimate."] if has_tax else ["- Tax rate not found per client rules."]
    client_lines = [
        "- Compared estimate to client guidelines (docs/photos/tax present where required).",
        "- Parts policy evaluated based on OEM-only vs aftermarket-preferred language."
    ]

    summary_md = "\n".join([
        "### Required Photos", *photo_lines,
        "### Labor Rates", *labor_lines,
        "### Taxes", *tax_lines,
        "### Parts Compliance", parts_summary,
        "### Client Rules Adherence", *client_lines
    ])

    # --- PDF
    pdf = FPDF(); pdf.add_page()
    try:
        pdf.add_font("DejaVu","","DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(0,10,"NSPXN.com AI Review Report", ln=True, align="C")
    pdf.set_font_size(10)
    pdf_kv(pdf, "File Number", file_number)
    pdf_kv(pdf, "IA Company", ia_company)
    pdf_kv(pdf, "Appraiser ID #", appraiser_id); pdf.ln(2)
    pdf_kv(pdf, "Claim #", claim)
    pdf_kv(pdf, "VIN", vin_est)
    pdf_kv(pdf, "VIN Photo Verification", vin_verify)
    pdf_kv(pdf, "Vehicle", vehicle)
    pdf_kv(pdf, "Compliance Score", f"{score}%")

    pdf.ln(4); pdf_h1(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0,6, f"**Audit Results: {score}%**")
    pdf.ln(1); pdf.multi_cell(0,6, summary_md)

    pdf.ln(4); pdf_h1(pdf, "Estimate ↔ Photos Consistency Review")
    if compare.get("per_item"):
        for it in compare["per_item"][:50]:
            ev = "YES" if it.get("photo_evidence") else "NO"
            conf = it.get("confidence", 0)
            try: conf = int(round(float(conf)*100))
            except Exception: conf = 0
            line = f"- {it.get('side','unspecified').title()} {it.get('part','component')} · {it.get('op','op')} → Photo: {ev} ({conf}%); {it.get('note','')}"
            pdf.multi_cell(0,6, line)
    else:
        pdf.multi_cell(0,6,"Per-item comparison unavailable.")

    if compare.get("not_in_photos"):
        pdf.ln(2); pdf_h1(pdf, "Items Estimated but Not Evident in Photos")
        for s in compare["not_in_photos"][:30]:
            pdf.multi_cell(0,6,f"- {s}")

    if compare.get("extra_damage_in_photos"):
        pdf.ln(2); pdf_h1(pdf, "Damage Visible in Photos but Missing on Estimate")
        for s in compare["extra_damage_in_photos"][:30]:
            pdf.multi_cell(0,6,f"- {s}")

    pdf.ln(2); pdf_kv(pdf, "Consistency Overall", compare.get("overall",""))

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
        with open(pdf_path, "wb") as fh:
            fh.write(pdf_bytes)
        log.info(f"PDF saved: {pdf_path}")
    except Exception as e:
        log.error(f"PDF save error: {e}")

    # --- Email (ALWAYS attach PDF)
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim}"
        msg["From"] = "info@nspxn.com"
        msg["To"] = "info@nspxn.com"
        body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Claim #: {claim}
VIN: {vin_est}
VIN Photo Verification: {vin_verify}
Vehicle: {vehicle}

Compliance Score: {score}%

AI Review Summary:
Audit Results: {score}%

{summary_md}
"""
        msg.set_content(body)
        try:
            with open(pdf_path, "rb") as fh:
                msg.add_attachment(fh.read(), maintype="application", subtype="pdf", filename=f"{file_number}.pdf")
        except Exception as e:
            log.error(f"Attach error: {e}")

        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        log.error(f"Email send error: {e}")

    return {
        "file_number": file_number,
        "claim_number": claim,
        "vehicle": vehicle,
        "vin": vin_est,
        "vin_photo_verification": vin_verify,
        "score": f"{score}%",
        "summary": summary_md,
        "consistency": compare
    }

@app.get("/download-pdf")
def download_pdf(file_number: str):
    path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})























