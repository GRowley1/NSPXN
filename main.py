from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI
import base64
import io
import os
from PyPDF2 import PdfReader
from docx import Document
import re
import smtplib
from email.message import EmailMessage
from fpdf import FPDF
from pdf2image import convert_from_bytes
import pytesseract

client = OpenAI()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_with_ocr(pdf_bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        return "\n".join([pytesseract.image_to_string(img) for img in images])
    except Exception:
        return ""

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_field(label, text):
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
    file_descriptions = []

    for file in files:
        content = await file.read()
        name = file.filename.lower()
        file_descriptions.append(name)

        if name.endswith((".jpg", ".jpeg", ".png")):
            b64 = base64.b64encode(content).decode("utf-8")
            images.append({
                "type": "image_url",
                "image_url": { "url": f"data:image/jpeg;base64,{b64}" }
            })
        elif name.endswith(".pdf"):
            text = extract_text_from_pdf(io.BytesIO(content))
            if not text.strip():
                text = extract_text_with_ocr(content)
            if "kbb" in name:
                text = "📄 KBB Private Party Valuation Included\n" + text
            elif "jd power" in name or "jdpower" in name:
                text = "📄 J.D. Power Valuation Document Included\n" + text
            texts.append(text)
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(content)))
        elif name.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {file.filename}")

    vision_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Uploaded files: {', '.join(file_descriptions)}"
            }
        ]
    }

    if texts:
        vision_message["content"].append({
            "type": "text",
            "text": "\n\n".join(texts)
        })
    if images:
        vision_message["content"].extend(images)

    prompt = f"""You are an AI auto damage auditor.
Compare the estimate against the damage photos and valuation documents.
If a KBB or JD Power valuation is present, reference it in your findings.
Always start your response with:
Claim #: (from estimate)
VIN: (from estimate or photos)
Vehicle: (make, model, mileage from estimate)
Compliance Score: (0–100%)

Then summarize findings and rule violations based on the following rules:
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
        vin = extract_field("VIN", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        score = extract_field("Compliance Score", gpt_output)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.ln(5)

        lines = gpt_output.splitlines()
        cleaned_lines = []
        seen_fields = {"claim": False, "vin": False, "vehicle": False, "score": False, "file": False}
        for line in lines:
            lowered = line.lower()
            if any(f in lowered for f in ["claim #", "vin", "vehicle", "compliance score", "file number"]):
                for key in seen_fields:
                    if key in lowered and not seen_fields[key]:
                        seen_fields[key] = True
                        cleaned_lines.append(line)
                        break
            else:
                cleaned_lines.append(line)
        cleaned_output = "\n".join(cleaned_lines)

        pdf.multi_cell(0, 10, f"AI4IA Review Summary:\n{cleaned_output}")
        pdf_path = f"{file_number}.pdf"
        pdf.output(pdf_path)

        # Email logic
        msg = EmailMessage()
        msg["Subject"] = f"AI4IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}

AI Review Summary:
{cleaned_output}
""")

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
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "gpt_output": "⚠️ AI review failed."}
        )

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type='application/pdf', filename=pdf_path)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})
