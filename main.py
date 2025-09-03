```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import re
import base64
import io
import smtplib
from email.message import EmailMessage
from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='/app/app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("\u274c OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com",
        "https://www.nspxn.com",
        "http://nspxn.com",
        "http://www.nspxn.com",
        "https://nspxn.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")  # Convert to grayscale
    img = ImageEnhance.Contrast(img).enhance(2.0)  # Enhance contrast
    img = img.filter(ImageFilter.MedianFilter(size=3))  # Noise reduction
    img = ImageOps.autocontrast(img)  # Adaptive thresholding
    img = ImageOps.invert(img)  # Invert for better OCR
    return img

def extract_text_from_pdf(file) -> str:
    try:
        file.seek(0)
        images = convert_from_bytes(file.read(), dpi=200)
        text_output = ""
        for i, img in enumerate(images, 1):
            processed = preprocess_image(img)
            try:
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 3')
            except Exception as e:
                logger.warning(f"PSM 3 failed for page {i}: {str(e)}, retrying with PSM 6")
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 6')
            if len(ocr_text.strip()) < 50 or re.search(r"[\\/:\\d\\s]{50,}", ocr_text):
                logger.warning(f"Page {i} OCR output skipped (garbled): {ocr_text[:100]}...")
                continue
            text_output += f"\n[Page {i}]\n{ocr_text}"
        if not text_output.strip():
            logger.error("No valid text extracted from PDF")
        return text_output
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return f"\n\u274c OCR error: {str(e)}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    logger.debug(f"Extracted DOCX text: {text[:500]}...")
    return text

def extract_field(label: str, text: str) -> str:
    if not label or not text:
        return "N/A"
    low_text = text.lower()
    low_label = label.lower()
    pos = low_text.find(low_label)
    if pos == -1:
        return "N/A"
    i = pos + len(label)
    while i < len(text) and text[i] in " \t:#=-":
        i += 1
    j = i
    while j < len(text) and text[j] not in "\r\n;":
        j += 1
    value = text[i:j].strip()
    if label.strip().lower() == "vin":
        m = re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", value, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
    return value or "N/A"

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...)
):
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error": "Appraiser ID is required."})

    images = []
    texts = []

    for file in files:
        content = await file.read()
        name = file.filename.lower()
        if name.endswith((".jpg", ".jpeg", ".png")):
            b64 = base64.b64encode(content).decode("utf-8")
            images.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        elif name.endswith(".pdf"):
            texts.append(extract_text_from_pdf(io.BytesIO(content)))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(content)))
        elif name.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {file.filename}")

    combined_text = '\n'.join(texts).lower()
    logger.debug(f"Combined text: {combined_text[:1000]}...")
    logger.debug(f"Client rules: {client_rules[:500]}...")

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({"type": "text", "text": '\n\n'.join(texts)})
    if images:
        vision_message["content"].extend(images)
    prompt = f"""
    You are an AI auto damage auditor. Your task is to compare damage photos to the estimate and client guidelines.

    Tasks:
    1. Extract key information: Claim #, VIN, Vehicle (make, model, mileage).
    2. Analyze if the damage photos match the damages listed in the estimate.
    3. Check compliance with client guidelines: {client_rules}
    4. Assign a Compliance Score (0–100%) based on the match between photos and estimate, and adherence to guidelines.
    5. Summarize findings, listing any mismatches or violations.

    At the top of your response, include:
    Claim #: (from estimate)
    VIN: (from estimate or photos)
    Vehicle: (make, model, mileage from estimate)
    Compliance Score: (0–100%)

    Summarize findings and rule violations below.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Fallback to a known model; replace with "gpt-5" if confirmed available
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_completion_tokens=3500
        )
        msg_obj = response.choices[0].message
        gpt_output = getattr(msg_obj, "content", None)
        if gpt_output is None:
            try:
                msg_dict = msg_obj.model_dump()
                gpt_output = msg_dict.get("content")
            except Exception:
                gpt_output = None
        if isinstance(gpt_output, list):
            parts = []
            for part in gpt_output:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            gpt_output = "".join(parts)
        if not gpt_output or not str(gpt_output).strip():
            logger.error("GPT returned no output or empty response")
            return JSONResponse(status_code=500, content={"error": "AI review failed: No output from GPT", "gpt_output": "⚠️ GPT returned no output."})
        logger.debug(f"GPT output: {gpt_output[:1000]}...")
        claim_number = extract_field("Claim #", gpt_output)
        vin = extract_field("VIN", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        score_text = extract_field("Compliance Score", gpt_output)

        try:
            score = int(score_text.strip("%"))
        except:
            score = 100
            logger.warning("Invalid Compliance Score format; defaulting to 100")

        logger.debug(f"Score from AI: {score}")

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "/app/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "AI Review Summary:", align='L')
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, gpt_output)

        pdf_path = f"/app/{file_number}.pdf"
        pdf.output(pdf_path)

        msg = EmailMessage()
        msg["Subject"] = f"AI Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Compliance Score: {score}%

AI Review Summary:
{gpt_output}
"""
        msg.set_content(email_body.encode("utf-8", errors="ignore").decode("utf-8"))
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)

        return {
            "gpt_output": gpt_output,
            "file_number": file_number,
            "claim_number": claim_number,
            "vehicle": vehicle,
            "score": f"{score}%"
        }

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"AI review failed: {str(e)}", "gpt_output": "⚠️ AI review failed."})

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"/app/{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = "/app/client_rules"
    file_name = f"{client_name}.docx"
    file_path = os.path.join(rules_dir, file_name)
    if os.path.exists(file_path):
        try:
            doc = Document(file_path)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            logger.debug(f"Client rules for {client_name}: {text[:500]}...")
            return {"text": text}
        except Exception as e:
            logger.error(f"Client rules error: {str(e)}")
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        logger.error(f"Rules not found for client: {client_name}")
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})
```