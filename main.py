from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI
import base64
import io
import os
import re
from PyPDF2 import PdfReader
from docx import Document
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

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"./pdfs/{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"error": "PDF not found"})

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

    vision_message = {
        "role": "user",
        "content": []
    }

    if texts:
        vision_message["content"].append({"type": "text", "text": "\n\n".join(texts)})
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

        os.makedirs("pdfs", exist_ok=True)
        pdf_path = f"./pdfs/{file_number}.pdf"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0)

        pdf.set_font(style="B", size=14)
        pdf.cell(200, 10, "NSPXN.com AI Review Report", ln=True, align="C")
        pdf.set_font(style="Arial", size=12)
        pdf.cell(200, 10, f"Date: {datetime.now().strftime('%B %d, %Y')}", ln=True)
        pdf.cell(200, 10, f"IA Company: {ia_company}", ln=True)

        pdf.ln(5)
        pdf.set_font(style="B", size=12)
        pdf.cell(200, 10, "AI Review Summary:", ln=True)
        pdf.set_font(style="Arial", size=12)
        pdf.multi_cell(0, 10, f"Claim #: {claim}\nVIN: {vin}\nVehicle: {vehicle}\nCompliance Score: {score}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, gpt_output)

        pdf.output(pdf_path)

        return {
            "gpt_output": gpt_output,
            "claim_number": claim,
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
