from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI
import base64
import io
import os
import re
import smtplib
from email.message import EmailMessage
from fpdf import FPDF
from docx import Document
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# ✅ Instantiate the app FIRST
app = FastAPI()

# ✅ CORS middleware NEXT
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

# ✅ Add the root route AFTER app declaration
@app.get("/")
async def root():
    return {"status": "ok"}
    
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF, fallback to OCR if pages contain images only."""
    reader = PdfReader(file)
    text_parts = []
    ocr_needed = False

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text_parts.append(page_text)
        else:
            ocr_needed = True

    combined_text = "\n".join(text_parts)

    if ocr_needed:
        try:
            file.seek(0)
            images = convert_from_bytes(file.read(), dpi=150, first_page=1, last_page=5)  # Limit pages
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                combined_text += ("\n" + ocr_text)
        except Exception as e:
            print(f"❌ OCR error: {str(e)}")
            combined_text += "⚠️ OCR failed for some pages."
    return combined_text

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_field(label, text) -> str:
    """Extract fields like Claim #, VIN, Vehicle, Compliance Score."""
    pattern = re.compile(rf"{label}[:\s-]*([^\n\r]+)", re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else "N/A"

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...),
    ia_company: str = Form(...)
):
    images = []
    texts = []
    for file in files:
        content = await file.read()
        name = file.filename.lower()

        if name.endswith((".jpg", ".jpeg", ".png")):
            b64 = base64.b64encode(content).decode("utf-8")
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        elif name.endswith(".pdf"):
            texts.append(extract_text_from_pdf(io.BytesIO(content)))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(content)))
        elif name.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {file.filename}")

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({
            "type": "text",
            "text": "\n\n".join(texts)
        })
    if images:
        vision_message["content"].extend(images)

    prompt = f"""
You are an AI auto damage auditor. You must review the uploaded estimate and associated documents (PDF, DOCX, images).
At the top of your response, always include:
Claim #: (from estimate)
VIN: (from estimate or photos)
Vehicle: (make, model, mileage from estimate)
Compliance Score: (0–100%)

Then summarize findings and rule violations based on the following rules:
{client_rules}

Important Clarification:
- Treat any mentions of "Description: Other (Add description to photo label)" or similar text as evidence that a photo was taken.
- Only mark photos as missing if NO mention (explicit or implicit) is found.
- Clearly state if photos are inferred from such lines and acknowledge their presence accordingly.
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
        vin = extract_field("VIN", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        score = extract_field("Compliance Score", gpt_output)

        # Create the PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "AI4IA Review Summary:", align='L')
        encoded_output = gpt_output.encode("latin-1", "replace").decode("latin-1")
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 10, encoded_output)

        pdf_path = f"{file_number}.pdf"
        pdf.output(pdf_path)

        # Send the email
        msg = EmailMessage()
        msg["Subject"] = f"AI4IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}

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
            "vin": vin,
            "vehicle": vehicle,
            "score": score
        }

    except Exception as e:
        print("❌ GPT Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "⚠️ AI review failed."})


@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type='application/pdf', filename=pdf_path)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})
