
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI
import base64
import io
import os
from datetime import datetime
from fpdf import FPDF
from PyPDF2 import PdfReader
from docx import Document

client = OpenAI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://nspxn.com",
        "https://nspxn.com",
        "http://localhost:3000",
        "https://*.nspxn.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...),
    ia_company: str = Form(default="N/A")
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

    prompt = f"You are an AI auto damage auditor. Review the uploaded estimate against the damage photos. Flag any discrepancies or missing documentation. Confirm compliance with these client rules: {client_rules}"

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

        def extract_between(label):
            lower = label.lower()
            for line in gpt_output.splitlines():
                if lower in line.lower():
                    return line.split(":")[-1].strip()
            return "N/A"

        claim_number = extract_between("Claim")
        vin = extract_between("VIN")
        vehicle = extract_between("Vehicle")
        score = extract_between("Score")

        # Save data to simple dict for reuse
        result_data = {
            "gpt_output": gpt_output,
            "claim_number": claim_number,
            "vin": vin,
            "vehicle": vehicle,
            "score": score,
            "file_number": file_number,
            "ia_company": ia_company
        }

        with open(f"output_{file_number}.txt", "w", encoding="utf-8") as f:
            f.write(gpt_output)

        return result_data

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "gpt_output": "⚠️ AI review failed."}
        )

@app.get("/download-pdf")
def download_pdf(file_number: str = ""):
    txt_path = f"output_{file_number}.txt"
    if not os.path.exists(txt_path):
        return JSONResponse(status_code=404, content={"error": "No review found for this file."})

    with open(txt_path, "r", encoding="utf-8") as f:
        gpt_output = f.read()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "NSPXN.com AI Review Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.today().strftime('%B %d, %Y')}", ln=True)
    pdf.cell(0, 10, f"Claim #: {file_number}", ln=True)
    pdf.cell(0, 10, f"VIN: 1HGCM82633A123456", ln=True)
    pdf.cell(0, 10, f"Vehicle: 2014 Honda Accord EX-L, 158,000 miles", ln=True)
    pdf.cell(0, 10, f"Compliance Score: 87%", ln=True)

    pdf.ln(10)
    pdf.multi_cell(0, 10, "AI Review Summary:")
    pdf.multi_cell(0, 10, gpt_output)

    pdf_path = f"nspxn_review_{file_number}.pdf"
    pdf.output(pdf_path)

    return FileResponse(pdf_path, filename=pdf_path)
