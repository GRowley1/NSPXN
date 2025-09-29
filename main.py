from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
import os, re, io, json, logging, base64, zipfile, time, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

from openai import OpenAI

# ----------- Config ----------
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ai4ia-slim")

OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
MODEL_PRIMARY = os.getenv("OAI_MODEL", "gpt-4o-mini")
MODEL_FALLBACK = "gpt-3.5-turbo"

if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

INTENTS = {
    "guidelines_only": "Guidelines -> Estimate (no photos)",
    "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
    "photos_only": "Photos Only: Compare to Estimate",
    "invoices_with_photos": "Supplement <-> Invoices (+ Photos)",
    "docs_checklist": "Documentation Checklist",
}

# ----------- App ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ----------- Tiny helpers ----------
def _pp(img):
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageOps.autocontrast(img)
    return img

def fast_pdf_text(pdf_bytes: bytes, limit_pages: int = 8) -> str:
    out = []
    try:
        rdr = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for i, pg in enumerate(rdr.pages[:limit_pages], 1):
            t = pg.extract_text() or ""
            if t.strip(): out.append(f"[Page {i}]\n{t}")
    except Exception as e:
        log.warning(f"PDF text read failed: {e}")
    return "\n\n".join(out)

def quick_pdf_ocr(pdf_bytes: bytes, max_pages: int = 2, dpi: int = 200) -> str:
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
        return "\n\n".join(
            f"[OCR {i+1}]\n{pytesseract.image_to_string(_pp(im), lang='eng', config='--psm 6')}"
            for i, im in enumerate(imgs)
        )
    except Exception as e:
        log.warning(f"OCR fallback failed: {e}")
        return ""

# VIN / vehicle extraction
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
VIN_TIGHT = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
VIN_RELAX = re.compile(r"(?:V\.?I\.?N\.?|Vehicle\s+Identification\s+Number|VIN)\b[^A-Z0-9]{0,40}((?:[A-HJ-NPR-Z0-9][\s\-]*){17})", re.IGNORECASE)
_trans = {**{str(i): i for i in range(10)}, **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_w = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def _norm_vin(s: str) -> Optional[str]:
    s = (s or "").upper()
    s = re.sub(r"[^A-HJ-NPR-Z0-9]", "", s).replace("O","0").replace("I","1").replace("Q","0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s): return None
    return s

def _vin_ok(v: str) -> bool:
    try:
        tot = sum(_trans[ch]*_w[i] for i,ch in enumerate(v)); chk = tot % 11
        return v[8] == ("X" if chk==10 else str(chk))
    except: return False

def vin_from_text(text: str) -> Optional[str]:
    cands = [m.group(1) for m in VIN_RELAX.finditer(text or "")] + VIN_TIGHT.findall(text or "")
    uniq=[]; seen=set()
    for c in cands:
        v = _norm_vin(c)
        if v and v not in seen: uniq.append(v); seen.add(v)
    for v in uniq:
        if _vin_ok(v): return v
    return uniq[0] if uniq else None

MAKE_MAP = {"NISSAN":"Nissan","CHEV":"Chevrolet","CHEVY":"Chevrolet","TOYOTA":"Toyota","FORD":"Ford","HONDA":"Honda","HYUNDAI":"Hyundai","KIA":"Kia","BMW":"BMW","MERCEDES":"Mercedes-Benz","MB":"Mercedes-Benz","VW":"Volkswagen","VOLKS":"Volkswagen","SUBARU":"Subaru","MAZDA":"Mazda","DODGE":"Dodge"}
STOP = {"GASOLINE","DIESEL","HYBRID","ELECTRIC","BLACK","WHITE","BLUE","RED","SILVER","GRAY","GREY","4D","2D","SED","SDN","SUV","COUPE","HATCH","TRUCK","WAGON","AWD","FWD","RWD","L","GDI","TURBO","PAINT","CLEAR","COAT","COLOR"}

def vehicle_from_text(text: str) -> Optional[str]:
    for ln in (text or "").splitlines():
        ln = re.sub(r"\s{2,}"," ",ln.strip())
        if re.match(r"^\s*(19|20)\d{2}\b", ln) and not re.search(r"\b(AM|PM)\b", ln):
            parts = ln.split(); year=parts[0]; tail=parts[1:]
            keep=[]
            for t in tail:
                raw = re.sub(r"[^\w\-]","",t).upper()
                if raw in STOP or raw in ("A/M","OEM"): break
                keep.append(t)
                if len(keep)>=4: break
            if keep:
                mk = MAKE_MAP.get(keep[0].upper(), keep[0].capitalize())
                return " ".join([year, mk]+keep[1:])
    return None

def mileage_from_text(text: str) -> Optional[str]:
    m = re.search(r"(?:Odometer|Mileage|Miles)\s*[:\-]?\s*([\d,]{2,7})\b", text or "", re.IGNORECASE)
    return m.group(1) if m else None

# Claim extractor that REQUIRES at least one digit (avoids “Services”)
def claim_from_text(text: str) -> Optional[str]:
    if not text: return None
    CLAIM_TOKEN = r"[A-Za-z0-9][A-Za-z0-9\-_\/]*\d[A-Za-z0-9\-_\/]*"
    pats = [
        rf"(?:Carrier|Insurance|Insurer)?\s*Claim\s*(?:No\.?|Number|#)?\s*[:\-]?\s*({CLAIM_TOKEN})",
        rf"(?:Assignment|Reference|Ref)\s*(?:No\.?|Number|#)?\s*[:\-]?\s*({CLAIM_TOKEN})",
        rf"(?<!Policy)\bClaim\b[^A-Za-z0-9]{{0,20}}({CLAIM_TOKEN})",
    ]
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".:,;")
    return None

def days_from_text(text: str) -> Optional[int]:
    m = re.search(r"Days?\s*to\s*Repair\s*[:\-]?\s*([0-9]+)", text or "", re.IGNORECASE)
    try: return int(m.group(1)) if m else None
    except: return None

def sanitize_latin1(s: str) -> str:
    if not s: return ""
    repl = {"\u2018":"'","\u2019":"'","\u201C":'"',"\u201D":'"',"\u2013":"-","\u2014":"-","\u2022":"-","\u2026":"...","\u2192":"->","\u2194":"<->","\u00A0":" "}
    for k,v in repl.items(): s = s.replace(k,v)
    return s.encode("latin-1","ignore").decode("latin-1","ignore")

def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.\-]+","-",s)
    return s.strip("-_.") or f"report-{int(time.time())}"

def final_percent(narr: str) -> Optional[int]:
    m = re.search(r"(final\s*(evaluation|score|compliance)\s*[:\-]?\s*)(\d{1,3})\s*%", narr or "", re.IGNORECASE)
    if m:
        try:
            v = int(m.group(3)); 
            return v if 0<=v<=100 else None
        except: return None
    return None

def vin_line_from_narrative(narr: str, photos_present: bool) -> str:
    if not photos_present:
        return "Photos not provided"
    if not narr:
        return "Included in narrative"
    m = re.search(r"VIN\s*Verification\s*:\s*(MATCH|MISMATCH|NOT\s*VERIFIED|PHOTOS\s*NOT\s*PROVIDED)", narr, re.IGNORECASE)
    if not m:
        return "Included in narrative"
    tag = m.group(1).upper().replace("  ", " ")
    if tag == "MATCH": return "Verified: MATCH"
    if tag == "MISMATCH": return "Verified: MISMATCH"
    if tag == "NOT VERIFIED": return "Not verified"
    return "Photos not provided"

def openai_chat(messages, max_tokens=800):
    for attempt in range(2):
        try:
            return client.chat.completions.create(
                model=MODEL_PRIMARY, messages=messages, max_tokens=max_tokens, temperature=0,
                timeout=OPENAI_TIMEOUT
            )
        except Exception as e:
            if "429" in str(e) or "timeout" in str(e).lower(): time.sleep(1.25*(attempt+1)); continue
            break
    try:
        return client.chat.completions.create(
            model=MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=0,
            timeout=OPENAI_TIMEOUT
        )
    except Exception as e:
        log.error(f"OAI fail: {e}"); return None

# ----------- API ----------
@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("guidelines_only")
):
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error":"Appraiser ID is required."})
    intent = ai_intent if ai_intent in INTENTS else "guidelines_only"
    request_type_label = INTENTS[intent]
    file_number = safe_filename(file_number)

    # Gather content
    pdfs: List[Tuple[str, bytes]] = []
    images: List[Tuple[str, bytes]] = []
    texts: List[str] = []

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            pdfs.append((name, raw))
        elif name.endswith((".jpg",".jpeg",".png",".webp")):
            images.append((name, raw))
        elif name.endswith(".docx"):
            try:
                txt = "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip())
                texts.append(txt)
            except: pass
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8","ignore"))
        elif name.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for zi in z.infolist():
                        if zi.is_dir(): continue
                        zname = zi.filename.lower(); zdata = z.read(zi)
                        if zname.endswith(".pdf"): pdfs.append((zname, zdata))
                        elif zname.endswith((".jpg",".jpeg",".png",".webp")): images.append((zname, zdata))
                        elif zname.endswith(".txt"): texts.append(zdata.decode("utf-8","ignore"))
            except Exception as e:
                log.warning(f"ZIP failed: {e}")

    photos_present = len(images) > 0

    # Prefer an estimate PDF containing 'est'
    est_pdf = None
    for nm, blob in pdfs:
        if "est" in nm or "estimate" in nm:
            est_pdf = (nm, blob); break
    if est_pdf is None and pdfs: est_pdf = pdfs[0]

    # Fast text; tiny OCR only if needed
    est_text = ""
    if est_pdf:
        est_text = fast_pdf_text(est_pdf[1], limit_pages=8)
        if len(est_text.strip()) < 40:
            est_text = quick_pdf_ocr(est_pdf[1], max_pages=2, dpi=200)
    if texts and not est_text:
        est_text = "\n\n".join(texts)[:12000]

    # Always extract these from the estimate
    vin = vin_from_text(est_text) or "N/A"
    vehicle = vehicle_from_text(est_text) or "N/A"
    mileage = mileage_from_text(est_text)
    claim = claim_from_text(est_text) or "N/A"
    days = days_from_text(est_text)

    # Build GPT messages strictly by the selected request
    sys_common = "You are an auto-claims appraisal assistant. Only analyze what the selected Request Type allows. Be concise, bullet-first, no fluff. Always end with a single line: 'Final Evaluation: NN%'."
    facts = {"photos_present": photos_present, "vin_estimate": vin, "vehicle": vehicle, "claim": claim, "mileage_present": bool(mileage)}
    messages = [{"role":"system","content":sys_common + " " + json.dumps(facts)}]

    def img_payload(max_imgs=12):
        out=[]
        for _, blob in images[:max_imgs]:
            b64 = base64.b64encode(blob).decode("utf-8")
            out.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
        return out

    if intent == "guidelines_only":
        system = "Analyze client guidelines strictly against the ESTIMATE only. If photos_present=false, do NOT mention photos."
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(client_rules or "")[:9000]},
            {"type":"text","text":"\n\nESTIMATE TEXT:\n"+(est_text or "")[:12000]}
        ]
        messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":user})
        rsp = openai_chat(messages, max_tokens=650)

    elif intent == "comprehensive":
        # Force the full structure so Comprehensive is never “missing a lot”
        system = (
            "Comprehensive review: guidelines vs estimate AND estimate vs photos. "
            "If photos_present=false, omit photo sections entirely. "
            "MANDATORY OUTPUT SHAPE:\n"
            "VIN Verification: <MATCH | MISMATCH | NOT VERIFIED | PHOTOS NOT PROVIDED>\n"
            "1) Client Quick Summary (2–3 bullets)\n"
            "2) Fatal Errors (bullet list)\n"
            "3) Client Photo Rules (only if photos_present=true) — each item begins with [Compliant] | [Non-compliant] | [Not found]\n"
            "4) Estimate/Supplement Release Rules — bracketed tags per item\n"
            "5) Parts Application Rules — bracketed tags per item\n"
            "6) Total Loss Rules — bracketed tags per item (or 'Not applicable')\n"
            "7) Tow Charge Rules — bracketed tags per item\n"
            "8) Supplement Handling Rules — bracketed tags per item\n"
            "9) Betterment/Depreciation Rules — bracketed tags per item\n"
            "10) Documentation Requirements — bracketed tags per item (call out Clean Retail Value & Advisor Report explicitly)\n"
            "11) Rates and Sales Tax Rules — bracketed tags per item\n"
            "12) Miscellaneous Rules — bracketed tags per item\n"
            "13) Estimate ↔ Photos Comparison (only if photos_present=true): damage match, discrepancies, missing views/measurements\n"
            "14) Summary & Next Steps (2 bullets)\n"
            "Always end with a single line: Final Evaluation: NN%."
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(client_rules or "")[:8000]},
            {"type":"text","text":"\n\nESTIMATE TEXT:\n"+(est_text or "")[:12000]}
        ]
        if photos_present: user += img_payload()
        messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":user})
        rsp = openai_chat(messages, max_tokens=1100)

    elif intent == "photos_only":
        if not photos_present:
            rsp = None
            gpt_output = "No photos were provided with this request."
        else:
            system = (
                "Compare photos to the estimate only. Do not restate guidelines. "
                "MANDATORY OUTPUT SHAPE:\n"
                "VIN Verification: <MATCH | MISMATCH | NOT VERIFIED | PHOTOS NOT PROVIDED>\n"
                "Photo Coverage\n"
                "Visible Damage vs Estimate\n"
                "Discrepancies\n"
                "Summary\n"
                "Final Evaluation: NN%."
            )
            user = [{"type":"text","text":"ESTIMATE TEXT:\n"+(est_text or "")[:9000]}] + img_payload()
            messages.append({"role":"system","content":system})
            messages.append({"role":"user","content":user})
            rsp = openai_chat(messages, max_tokens=800)

    elif intent == "invoices_with_photos":
        inv_text = ""
        for nm, blob in pdfs:
            if est_pdf and nm == est_pdf[0]: continue
            if any(k in nm for k in ["invoice","supplement","receipt"]):
                inv_text += fast_pdf_text(blob, limit_pages=4)
        system = "Compare supplement/invoices to the estimate, and (if present) to photos. Sections: Invoices Summary, Support vs Estimate Lines, Photo Corroboration (if photos_present), Missing Documentation, Summary. End with 'Final Evaluation: NN%'."
        user = [
            {"type":"text","text":"ESTIMATE TEXT:\n"+(est_text or "")[:8000]},
            {"type":"text","text":"\n\nINVOICES TEXT:\n"+(inv_text or '')[:6000]}
        ]
        if photos_present: user += img_payload()
        messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":user})
        rsp = openai_chat(messages, max_tokens=900)

    elif intent == "docs_checklist":
        system = "Documentation checklist only. State present/missing for each item required by client guidelines based on the estimate text. End with 'Final Evaluation: NN%'."
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(client_rules or "")[:6000]},
            {"type":"text","text":"\n\nESTIMATE TEXT:\n"+(est_text or "")[:8000]}
        ]
        messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":user})
        rsp = openai_chat(messages, max_tokens=600)

    gpt_output = (rsp.choices[0].message.content if rsp else locals().get("gpt_output","Automated narrative unavailable.")).strip()
    if not photos_present:
        gpt_output = re.sub(r"(?im)^.*photo.*$", "", gpt_output).strip()

    comp = final_percent(gpt_output)

    # VIN verification line in header (mirrors narrative, no image OCR)
    if intent == "comprehensive":
        vin_line = vin_line_from_narrative(gpt_output, photos_present)
    elif intent in ("photos_only", "invoices_with_photos"):
        vin_line = "Included in narrative" if photos_present else "Not requested"
    else:
        vin_line = "Not requested"

    # ----------- PDF (same format) ----------
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=11)
    pdf.cell(200,10,sanitize_latin1("NSPXN.com AI Review Report"),ln=True,align="C")
    pdf.ln(5); pdf.set_font_size(10)
    pdf.multi_cell(0,6,sanitize_latin1(f"File Number: {file_number}"))
    pdf.multi_cell(0,6,sanitize_latin1(f"IA Company: {ia_company}"))
    pdf.multi_cell(0,6,sanitize_latin1(f"Request Type: {request_type_label}"))
    pdf.multi_cell(0,6,sanitize_latin1(f"Appraiser ID #: {appraiser_id}"))
    pdf.ln(4)
    pdf.multi_cell(0,6,sanitize_latin1(f"Claim #: {claim or 'N/A'}"))
    pdf.multi_cell(0,6,sanitize_latin1(f"VIN (from estimate): {vin}"))
    pdf.multi_cell(0,6,sanitize_latin1(f"VIN verification (estimate vs photo): {vin_line}"))
    pdf.multi_cell(0,6,sanitize_latin1(f"Vehicle: {vehicle}"))
    if mileage: pdf.multi_cell(0,6,sanitize_latin1(f"Odometer (from estimate): {mileage}"))
    if days is not None: pdf.multi_cell(0,6,sanitize_latin1(f"Days to Repair (reported): {days}"))
    pdf.multi_cell(0,6,sanitize_latin1(f"Compliance Score: {comp if comp is not None else 'N/A'}%"))
    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0,8,sanitize_latin1("AI-4-IA Review Summary"),ln=True)
    pdf.set_font_size(10); pdf.multi_cell(0,6,sanitize_latin1(gpt_output or "No narrative generated."))
    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0,8,sanitize_latin1("Estimate <-> Photos Consistency Review"),ln=True)
    pdf.set_font_size(10)
    pdf.multi_cell(0,6,sanitize_latin1("Included in narrative above.") if (intent in ("comprehensive","photos_only","invoices_with_photos") and photos_present) else sanitize_latin1("Not requested or no photos provided."))

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    with open(pdf_path,"wb") as f: f.write(pdf_bytes)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_url = f"/download-pdf?file_number={file_number}"

    # ----------- Email (same shell) ----------
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim or 'N/A'}"
        msg["From"] = "noreply@nspxn.com"; msg["To"] = "info@nspxn.com"
        body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Request Type: {request_type_label}

Claim #: {claim or 'N/A'}
VIN (from estimate): {vin}
VIN verification (estimate vs photo): {vin_line}
Vehicle: {vehicle}
{('Odometer (from estimate): ' + str(mileage)) if mileage else ''}
{('Days to Repair (reported): ' + str(days)) if days is not None else ''}

Compliance Score: {(str(comp)+'%') if comp is not None else 'N/A'}

Summary:
{gpt_output}
"""
        msg.set_content(body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        log.warning(f"Email send error (continuing): {e}")

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
        "vin_verification": vin_line,
        "odometer_estimate": mileage or "Not documented",
        "days_to_repair": days or "Not documented",
        "compliance_score": comp if comp is not None else "N/A",
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    safe = re.sub(r"[^\w.\-]+","-", file_number).strip("-_.")
    path = os.path.join(PDF_DIR, f"{safe}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{safe}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})










