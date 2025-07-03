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

def adjust_score(text: str, client_rules: str, initial_score: int) -> int:
    # Drop to 0% if labor hours exist and labor rate is $0.00
    if re.search(r"labor hours[:\s]*\d+", text, re.IGNORECASE):
        if re.search(r"labor rate[:\s]*\$?0+(\.00)?", text, re.IGNORECASE):
            return 0

    # Deduct 50% if tax required in client rules and no tax found
    if re.search(r"require.*tax", client_rules, re.IGNORECASE):
        if not re.search(r"tax[:\s]*\$?\d+|\d+%", text, re.IGNORECASE):
            return max(0, initial_score - 50)

    return initial_score

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
            texts.append(f"\u26a0\ufe0f Skipped unsupported file: {file.filename}")

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({"type": "text", "text": '\n\n'.join(texts)})
    if images:
        vision_message["content"].extend(images)

    prompt = f"""
You are an AI auto damage auditor. You have access to both text and images (or scans).

IMPORTANT RULES:
- Treat mentions of \"Clean Retail Value\" or \"NADA Value\" or \"Estimated Trade-In Value\" or \"Fair Market Range\" in the text as CONFIRMATION that the required Clean Retail Value printout was included.
- Treat mentions of \"CCC Advisor Report\" in the text as CONFIRMATION that the required Advisor Report printout was included.
- DO NOT mark photos as missing if any of the following conditions are met:
   - The label appears in the text
   - A visual appears in the uploaded documents
   - The text mentions CCC Advisor, which confirms inclusion of Advisor Report.
- Do NOT claim the \"Clean Retail Value\" is missing if text mentions its presence.
- Do NOT claim the \"Advisor Report\" is missing if text mentions its presence.
- Acknowledge evidence as present if indicated by labels, text, or actual uploaded images.

Perform a thorough review comparing the estimate against the damage photos and text. At the top of your response, ALWAYS include:
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
        gpt_output = response.choices[0].message.content or "\u26a0\ufe0f GPT returned no output."
        claim_number = extract_field("Claim", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        score = extract_field("Compliance Score", gpt_output)

        try:
            score_val = int(score.strip('%'))
        except:
            score_val = 100

        combined_text = '\n'.join(texts).lower()
        score_val = adjust_score(combined_text, client_rules, score_val)

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.multi_cell(0, 10, f"Final Adjusted Compliance Score: {score_val}%")
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
Adjusted Compliance Score: {score_val}%

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
            "score": f"{score_val}%"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "\u26a0\ufe0f AI review failed."})

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







