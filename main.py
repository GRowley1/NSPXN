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
from PIL import Image, ImageEnhance, ImageOps

from openai import OpenAI

# ----------- Config ----------
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ai4ia-core")

OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
MODEL_PRIMARY = os.getenv("OAI_MODEL", "gpt-4o-mini")
MODEL_FALLBACK = "gpt-3.5-turbo"

api_key = os.environ.get("OPENAI_API_KEY","")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=api_key)

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

# ----------- Helpers ----------
def _pp(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.7)
    img = ImageOps.autocontrast(img)
    return img

def fast_pdf_text(pdf_bytes: bytes, limit_pages: int = 6) -> str:
    out = []
    try:
        rdr = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for i, pg in enumerate(rdr.pages[:limit_pages], 1):
            t = pg.extract_text() or ""
            if t.strip(): out.append(f"[Page {i}]\n{t}")
    except Exception as e:
        log.warning(f"PDF text read failed: {e}")
    return "\n\n".join(out)

def first_page_ocr(pdf_bytes: bytes, dpi: int = 300) -> str:
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)[:1]
        if not imgs: return ""
        return "[OCR 1]\n" + pytesseract.image_to_string(_pp(imgs[0]), lang="eng", config="--psm 6")
    except Exception as e:
        log.warning(f"Header OCR failed: {e}")
        return ""

# --- Field extraction ---
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

# Claim extractor with stronger plausibility filter (avoid model text like 'sedan-4d')
MODEL_WORDS = re.compile(r"(sedan|coupe|hatch|suv|truck|van|wagon|convertible|"
                         r"corolla|altima|civic|accord|camry|sentra|elantra|rogue|rav4|tacoma|silverado|ram|f150)", re.I)

def _is_plausible_claim(tok: str) -> bool:
    if not tok: return False
    tok = tok.strip().strip(".:,;")
    if not (5 <= len(tok) <= 25): return False
    if len(re.findall(r"\d", tok)) < 2: return False   # force ≥2 digits
    if "/" in tok and MODEL_WORDS.search(tok): return False
    return True

def claim_from_text(text: str) -> Optional[str]:
    if not text: return None
    CLAIM_TOKEN = r"[A-Za-z0-9][A-Za-z0-9\-_\/]*\d[A-Za-z0-9\-_\/]*"
    labels = r"(?:Claim|Assignment|Reference|Ref|File|Loss|Case|Report|RO|Work\s*Order)"
    pats = [
        rf"(?:Carrier|Insurance|Insurer)?\s*{labels}\s*(?:No\.?|Number|#)?\s*[:\-]?\s*({CLAIM_TOKEN})",
        rf"(?<!Policy)\b{labels}\b[^A-Za-z0-9]{{0,20}}({CLAIM_TOKEN})",
    ]
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cand = m.group(1)
            if _is_plausible_claim(cand):
                return cand.strip().rstrip(".:,;")
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

def parse_vin_verification(narr: str) -> Optional[str]:
    m = re.search(r"VIN\s*Verification\s*:\s*(MATCH|MISMATCH|NOT\s*VERIFIED|PHOTOS\s*NOT\s*PROVIDED)", narr or "", re.IGNORECASE)
    return m.group(1).upper().replace("  "," ") if m else None

def strip_photo_sections(narr: str) -> str:
    s = narr
    s = re.sub(r'(?is)\n\s*Estimate\s*[^\n]*Photos.*?(?=\n\s*[A-Z0-9]+[\).\s]|$)', '\n', s)
    s = re.sub(r'(?im)^.*photo.*$', '', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def openai_chat(messages, max_tokens=900):
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

# --- Valuation detection (ANY source) ---
VAL_KEYS = [
    ("Kelley Blue Book", "KBB"),
    ("KBB.com", "KBB"),
    ("NADA", "NADA/J.D. Power"),
    ("J.D. Power", "NADA/J.D. Power"),
    ("Black Book", "Black Book"),
    ("Edmunds", "Edmunds"),
]
GENERIC_PAT = re.compile(r"(Average Price|Estimated Trade-?In Value|Clean Retail Value|Vehicle Valuation)", re.I)

def detect_valuations(pdfs: List[Tuple[str, bytes]], est_pdf_name: Optional[str]) -> Tuple[bool, list, list]:
    sources, names = [], []
    for nm, blob in pdfs:
        if est_pdf_name and nm == est_pdf_name:
            continue
        try:
            t = (first_page_ocr(blob) + "\n" + fast_pdf_text(blob, limit_pages=2)).upper()
        except:
            t = ""
        hit = None
        for key, lab in VAL_KEYS:
            if key.upper() in t:
                hit = lab; break
        if not hit and GENERIC_PAT.search(t):
            hit = "Generic Valuation"
        if hit:
            sources.append(hit)
            names.append(nm)
    sources = list(dict.fromkeys(sources))
    return (len(sources) > 0), sources, names

# --- Client rules detection (from uploaded files) ---
RULE_NAME_HINT = re.compile(r"(rule|guideline|policy|client|procedur|fatal\s+error|photo\s+rules|quick\s+summary)", re.I)
RULE_TEXT_HINT = re.compile(
    r"(quick\s+summary|fatal\s+errors?|photo\s+rules|parts\s+application|supplement|documentation\s+requirements|rates?\s*&\s*tax|tow\s+charge|total\s+loss)",
    re.I
)

def detect_client_rules(pdfs: List[Tuple[str, bytes]], texts: List[str], est_pdf_name: Optional[str]) -> Tuple[bool, str, list]:
    collected = []
    sources = []
    # TXT / DOCX texts already extracted in `texts`
    for t in texts:
        if RULE_TEXT_HINT.search(t):
            collected.append(t)
            sources.append("text-upload")
    # PDFs that look like rules (exclude the estimate)
    for nm, blob in pdfs:
        if est_pdf_name and nm == est_pdf_name:
            continue
        is_name_hit = RULE_NAME_HINT.search(nm or "")
        try:
            t = (first_page_ocr(blob) + "\n" + fast_pdf_text(blob, limit_pages=4))
        except:
            t = ""
        is_text_hit = RULE_TEXT_HINT.search(t or "")
        if is_name_hit or is_text_hit:
            if (t and len(t) > 400) or is_name_hit:
                collected.append(t)
                sources.append(nm)
    if collected:
        # Keep it to a sane size for the prompt
        joined = "\n\n".join(collected)
        return True, joined[:18000], sources[:6]
    return False, "", []

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

    # Collect inputs
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

    # Identify estimate PDF
    est_pdf = None
    for nm, blob in pdfs:
        if "est" in nm or "estimate" in nm:
            est_pdf = (nm, blob); break
    if est_pdf is None and pdfs: est_pdf = pdfs[0]

    # Extract estimate text
    est_text = ""
    if est_pdf:
        est_text = (first_page_ocr(est_pdf[1]) + "\n\n" + fast_pdf_text(est_pdf[1], limit_pages=6)).strip()

    # Combine other docs text
    all_pdf_text = est_text
    for nm, blob in pdfs:
        if not est_pdf or nm != est_pdf[0]:
            all_pdf_text += "\n\n" + first_page_ocr(blob) + "\n\n" + fast_pdf_text(blob, limit_pages=2)
    if texts:
        all_pdf_text += "\n\n" + "\n\n".join(texts)[:8000]

    # Parse header fields
    vin = vin_from_text(est_text) or vin_from_text(all_pdf_text) or "N/A"
    vehicle = vehicle_from_text(est_text) or vehicle_from_text(all_pdf_text) or "N/A"
    mileage = mileage_from_text(est_text) or mileage_from_text(all_pdf_text)
    claim = claim_from_text(est_text) or claim_from_text(all_pdf_text) or "N/A"
    days = days_from_text(est_text) or days_from_text(all_pdf_text)

    # Detect ANY valuation
    est_name = est_pdf[0] if est_pdf else None
    valuation_present, valuation_sources, valuation_doc_names = detect_valuations(pdfs, est_name)

    # Detect client rules from uploads if textarea is empty/short
    rules_text = (client_rules or "").strip()
    rules_present = len(rules_text) >= 50
    rules_sources = []
    if not rules_present:
        found, det_text, sources = detect_client_rules(pdfs, texts, est_name)
        if found:
            rules_text = det_text
            rules_present = True
            rules_sources = sources

    # GPT messages
    facts = {
        "photos_present": photos_present,
        "vin_estimate": vin,
        "vehicle": vehicle,
        "claim": claim,
        "mileage_estimate": mileage or "",
        "days_to_repair_estimate": days if days is not None else "",
        "valuation_present": valuation_present,
        "valuation_sources": valuation_sources,
        "rules_present": rules_present,
        "rules_sources": rules_sources,
        "intent": intent
    }
    sys_common = (
        "You are an auto-claims appraisal assistant. Use ONLY the provided materials (estimate text, uploaded photos, client rules). "
        "Do NOT invent details. If something is not present, write 'Not found in provided documents'. "
        "Total-loss-only items (salvage bids, valuation sheets, owner-retain) appear ONLY if the estimate explicitly declares a total loss; otherwise: Not applicable (repairable). "
        "For any 'Clean Retail Value' requirement: treat ANY uploaded valuation (KBB, NADA/J.D. Power, Black Book, Edmunds, generic valuation sheet) as compliant; name the source(s) you detect. "
        "VIN policy: only consider VINs that are 17 contiguous characters using A–H, J–N, P, R–Z and digits 0–9 (no I/O/Q). Normalize by removing spaces/hyphens; ignore anything not exactly 17 after normalization. "
        "Always end with a single line: Final Evaluation: NN%."
    )
    header_hint = {k:v for k,v in facts.items() if k in ["vin_estimate","vehicle","claim","mileage_estimate","days_to_repair_estimate","valuation_present","valuation_sources","rules_present","rules_sources"]}
    messages = [{"role":"system","content":sys_common + " Parsed header context: " + json.dumps(header_hint)}]

    def img_payload(max_imgs=12):
        out=[]
        for _, blob in images[:max_imgs]:
            b64 = base64.b64encode(blob).decode("utf-8")
            out.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
        return out

    if intent == "guidelines_only":
        system = (
            "Compare client guidelines to the ESTIMATE only. "
            "Do NOT restate guidelines verbatim. Produce compliance vs deviation with short evidence quotes from the estimate text. "
            "If rules_present=false, explicitly say: 'No client rules provided' and do not fabricate rules. "
            "Count ANY uploaded valuation as satisfying 'Clean Retail Value' and name the source(s) if present. "
            "Include an 'Estimate Snapshot' with totals, labor/paint materials hours & rates, tax rate, and days to repair—only if present in the estimate text. "
            "Mark truly missing items as 'Not found in provided documents'."
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(rules_text if rules_present else "")[:9000]},
            {"type":"text","text":"\n\nESTIMATE TEXT:\n"+(est_text or "")[:12000]}
        ]
        messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":user})
        rsp = openai_chat(messages, max_tokens=700)

    elif intent == "comprehensive":
        system = (
            "Comprehensive audit. Compare client guidelines ↔ estimate AND estimate ↔ photos. "
            "Never restate guidelines; list where the estimate COMPLIES vs DEVIATES, each with a 3–12 word evidence quote from the estimate text only. "
            "If rules_present=false, say 'No client rules provided' and skip compliance scoring vs rules (still do estimate↔photos). "
            "If photos_present=false, omit all photo sections AND do not refer to photos anywhere. "
            "When photos are provided, identify primary damage location(s) using automotive convention (Left/Right is driver-perspective); call out specific panels (e.g., left front door, left quarter, rear bumper). "
            "For 'Clean Retail Value', count ANY uploaded valuation as compliant; name the source(s). "
            "VIN Verification: output one of <MATCH | MISMATCH | NOT VERIFIED | PHOTOS NOT PROVIDED>. "
            "- MATCH only if an explicit 17-char estimate VIN equals a clear 17-char VIN from a VIN photo (after normalization). "
            "- MISMATCH only if both are explicit 17-char VINs and they differ. "
            "- PHOTOS NOT PROVIDED if no VIN photo exists. "
            "- NOT VERIFIED if a VIN photo exists but is unreadable/partial. "
            "Sections:\n"
            "- Estimate Snapshot (estimate text only): totals, labor/paint materials hours & rates, tax rate, days to repair (only if present)\n"
            "- VIN Verification: <...>\n"
            "- Compliance vs Guidelines (only if rules_present=true): [Compliant]/[Non-compliant]/[Not found] + short evidence quote\n"
            "- Estimate ↔ Photos: damage match (note LEFT/RIGHT & panels), discrepancies, missing photo angles/measurements\n"
            "- Missing Documents & Risks\n"
            "- Next Steps (1–3 bullets)\n"
            "Always end with: Final Evaluation: NN%."
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(rules_text if rules_present else "")[:8000]},
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
                "Compare photos to the estimate only. "
                "Do NOT restate guidelines. "
                "If totals are cited, include a brief 'Estimate Snapshot' (only if present in estimate text). "
                "Identify primary damage location(s) with left/right/panel naming. "
                "Sections: Photo Coverage, Visible Damage vs Estimate, Discrepancies, Summary. "
                "End with: Final Evaluation: NN%."
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
                inv_text += first_page_ocr(blob) + "\n" + fast_pdf_text(blob, limit_pages=2)
        system = (
            "Analyze supplement/invoices against the estimate (and photos if provided). "
            "Do NOT restate guidelines. "
            "Sections: Invoices Summary, Support vs Estimate Lines, Photo Corroboration (if photos_present; identify left/right/panels), Missing Documentation, Summary. "
            "End with: Final Evaluation: NN%."
        )
        user = [
            {"type":"text","text":"ESTIMATE TEXT:\n"+(est_text or "")[:8000]},
            {"type":"text","text":"\n\nINVOICES TEXT:\n"+(inv_text or '')[:6000]}
        ]
        if photos_present: user += img_payload()
        messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":user})
        rsp = openai_chat(messages, max_tokens=900)

    elif intent == "docs_checklist":
        system = (
            "Documentation checklist only based on estimate text (and photos if provided). "
            "If rules_present=false, say 'No client rules provided' and check only general documentation from the estimate/photos. "
            "Mark Present / Missing / Not found. End with: Final Evaluation: NN%."
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(rules_text if rules_present else "")[:6000]},
            {"type":"text","text":"\n\nESTIMATE TEXT:\n"+(est_text or "")[:8000]}
        ]
        if photos_present: user += img_payload(6)
        messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":user})
        rsp = openai_chat(messages, max_tokens=600)

    gpt_output = (rsp.choices[0].message.content if rsp else locals().get("gpt_output","Automated narrative unavailable.")).strip()

    if not photos_present and intent in ("comprehensive","photos_only","invoices_with_photos"):
        gpt_output = strip_photo_sections(gpt_output)

    comp = final_percent(gpt_output)

    vin_ver_tag = parse_vin_verification(gpt_output)
    if intent in ("comprehensive","photos_only","invoices_with_photos"):
        vin_line = (vin_ver_tag.capitalize() if vin_ver_tag
                    else ("Photos not provided" if not photos_present else "Not verified"))
    else:
        vin_line = "Not requested"

    # ----------- PDF ----------
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

    # ----------- Email ----------
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
        smtp_pass = os.getenv("NSPXN_SMTP_PASS", "grr2025GRR")
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", smtp_pass)
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
        "valuation_present": valuation_present,
        "valuation_sources": valuation_sources,
        "rules_present": rules_present,
        "rules_sources": rules_sources,
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    safe = re.sub(r"[^\w.\-]+","-", file_number).strip("-_.")
    path = os.path.join(PDF_DIR, f"{safe}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{safe}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})


