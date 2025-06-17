import csv
from datetime import datetime

def log_submission(claim, vin, vehicle, score, ia_company, file_number):
    with open("submissions.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.utcnow().isoformat(),
            file_number,
            ia_company,
            claim,
            vin,
            vehicle,
            score
        ])


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
from fpdf import FPDF
from datetime import datetime

client = OpenAI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in pdf.pages)

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

    for file in files:
        content = await file.read()
        name = file.filename.lower()

        if name.endswith((".jpg", ".jpeg", ".png")):
            b64 = base64.b64encode(content).decode("utf-8")
            images.append({
                "type": "image_url",
                "image_url": { "url": f"data:image/jpeg;base64,{b64}" }
            })
        elif name.endswith(".pdf"):
            texts.append(extract_text_from_pdf(io.BytesIO(content)))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(content)))
        elif name.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {file.filename}")

    vision_message = {
        "role": "user",
        "content": []
    }

    if texts:
        vision_message["content"].append({
            "type": "text",
            "text": "\n\n".join(texts)
        })
    if images:
        vision_message["content"].extend(images)

    prompt = f"""You are an AI auto damage auditor.
Compare the estimate against the damage photos.
At the top of your response, always include:
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

        claim = extract_field("Claim", gpt_output)
        vin = extract_field("VIN", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        score = extract_field("Compliance Score", gpt_output)

        # Save PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_title("NSPXN.com AI Review Report")

        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, "NSPXN.com AI Review Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Date: {datetime.today().strftime('%B %d, %Y')}", ln=True)
        pdf.cell(200, 10, f"IA Company: {ia_company}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, "AI Review Summary:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, gpt_output)

        filepath = os.path.join(REPORTS_DIR, f"{file_number}.pdf")
        pdf.output(filepath)

        log_submission(
            extract_field("Claim", gpt_output),
            extract_field("VIN", gpt_output),
            extract_field("Vehicle", gpt_output),
            extract_field("Compliance Score", gpt_output),
            ia_company,
            file_number
        )

        return {
            "gpt_output": gpt_output,
            "claim_number": claim,
            "vin": vin,
            "vehicle": vehicle,
            "score": score
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "gpt_output": "⚠️ AI review failed."
        })

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    filepath = os.path.join(REPORTS_DIR, f"{file_number}.pdf")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"error": "PDF not found"})
