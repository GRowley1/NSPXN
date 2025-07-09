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
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
from openai import OpenAI

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

def extract_text_from_pdf(file) -> str:
    try:
        file.seek(0)
        images = convert_from_bytes(file.read(), dpi=200)
        text_output = ""
        for i, img in enumerate(images, 1):
            img = img.convert("RGB")
            ocr_text = pytesseract.image_to_string(img, lang="eng")
            text_output += f"\n[Page {i}]\n" + ocr_text
        return text_output
    except Exception as e:
        return f"\n\u274c OCR error during combined extraction: {str(e)}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return '\n'.join(p.text for p in doc.paragraphs)

def extract_field(label, text) -> str:
    pattern = re.compile(rf"{label}[:\s-]*([^\n\r]+)", re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else "N/A"

def advisor_report_present(texts: List[str], image_files: List[UploadFile]) -> bool:
    for t in texts:
        if "ccc advisor report" in t.lower():
            return True
    for img in image_files:
        if "advisor" in img.filename.lower():
            return True
        ocr = pytesseract.image_to_string(Image.open(io.BytesIO(img.file.read())), lang="eng")
        if "advisor report" in ocr.lower():
            return True
    return False

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    labor_hours = re.search(r"(body|paint) labor\s+\d+\.?\d*\s+hrs", text, re.IGNORECASE)
    zero_rate = re.search(r"\$\s*0+(\.00)?\s*/hr", text, re.IGNORECASE)
    if labor_hours and zero_rate:
        return -100

    if re.search(r"utilize applicable tax rate", client_rules, re.IGNORECASE):
        if not re.search(r"tax[:\s]*\$?\d+|\d+%", text, re.IGNORECASE):
            return -75

    return 0

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
    image_files = []

    for file in files:
        content = await file.read()
        name = file.filename.lower()
        if name.endswith((".jpg", ".jpeg", ".png")):
            image_files.append(file)
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
    advisor_confirmed = advisor_report_present(texts, image_files)
    advisor_hint = "\n\nCONFIRMED: CCC Advisor Report is included based on OCR or filename." if advisor_confirmed else ""

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({"type": "text", "text": '\n\n'.join(texts) + advisor_hint})
    if images:
        vision_message["content"].extend(images)

    prompt = f"""
    You are an AI auto damage auditor. You have access to both text and images (or scans).

    IMPORTANT RULES:
    - If labor hours are present but the labor rate is $0 or missing, the Compliance Score must be set to 0%.
    - If tax is required by client rules but no tax rate is found in the estimate, reduce the Compliance Score by 75%.
    - Never assume compliance if required elements (like labor rate or taxes) are missing.
    - Treat mentions or OCR detection of \"Clean Retail Value\", or \"NADA Value\", or \"Fair Market Range\", or \"Estimated Trade-In Value\", as CONFIRMATION that the value was included.
    - Treat mentions or OCR detection of \"CCC Advisor Report\" as CONFIRMATION that the Advisor Report was included.
    - Do NOT rely on assumptions. Only acknowledge presence of documents or data when clearly present in text or visible in photos.
    - Only evaluate Total Loss protocols if the estimate or documentation explicitly indicates the vehicle was a total loss.
    - Do not assume a total loss condition based on estimate formatting or value alone.
    - If no mention of Total Loss or salvage is found, do not apply deductions for missing Total Loss evaluation details.

    PHOTO EVIDENCE RULES (override label dependency):
    [...continue prompt as before...]

    At the top of your response, ALWAYS include:
    Claim #: (from estimate)
    VIN: (from estimate or photos)
    Vehicle: (make, model, mileage from estimate)
    Compliance Score: (0–100%)

    Then summarize findings and rule violations based STRICTLY on the following rules:
    {client_rules}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                vision_message
            ],
            max_tokens=3500
        )
        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."
        claim_number = extract_field("Claim", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        score = extract_field("Compliance Score", gpt_output)

        try:
            score = int(score.strip("%"))
        except:
            score = 100

        score_adj = check_labor_and_tax_score(combined_text, client_rules)
        score = max(0, score + score_adj) if score_adj > -100 else 0

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.multi_cell(0, 10, f"AI-4-IA Final Compliance Score: {score}%")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "AI-4-IA Review Summary:", align='L')
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, gpt_output)

        pdf_path = f"{file_number}.pdf"
        pdf.output(pdf_path)

        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Adjusted Compliance Score: {score}%

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
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "⚠️ AI review failed."})

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=pdf_path)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = "client_rules"
    file_name = f"{client_name}.docx"
    file_path = os.path.join(rules_dir, file_name)
    if os.path.exists(file_path):
        try:
            doc = Document(file_path)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            return {"text": text}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})

