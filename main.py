
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
from fpdf import FPDF

client = OpenAI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://nspxn.com", "https://nspxn.com", "http://localhost:3000", "https://*.nspxn.com"
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

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"./pdfs/{file_number}.pdf"
    if not os.path.exists(pdf_path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})
    return FileResponse(path=pdf_path, filename=f"{file_number}.pdf", media_type="application/pdf")

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

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({"type": "text", "text": "\n\n".join(texts)})
    if images:
        vision_message["content"].extend(images)

    prompt = f'''
You are an AI auto damage auditor. Compare the damage estimate against the attached vehicle photos.

Respond ONLY in this format:
Claim Number: (value)
VIN: (value)
Vehicle: (value)
Compliance Score: (0-100)
Review Summary:
- Bullet point list of discrepancies or issues.

Client Rules:
{client_rules}
'''

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

        def extract_value(label):
            for line in gpt_output.splitlines():
                if label.lower() in line.lower():
                    return line.split(":", 1)[-1].strip()
            return "N/A"

        claim_number = extract_value("Claim")
        vin = extract_value("VIN")
        vehicle = extract_value("Vehicle")
        score = extract_value("Score")

        logo_path = f"./logos/{ia_company.lower()}.png"
        pdf = FPDF()
        pdf.add_page()
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=8, w=50)
        pdf.set_font("Arial", size=10)
        pdf.ln(40)
        pdf.multi_cell(0, 10, f"File Number: {file_number}\nClaim Number: {claim_number}\nVIN: {vin}\nVehicle: {vehicle}\nCompliance Score: {score}\n\nGPT Output:\n{gpt_output}")

        os.makedirs("./pdfs", exist_ok=True)
        pdf_path = f"./pdfs/{file_number}.pdf"
        pdf.output(pdf_path)

        return {
            "gpt_output": gpt_output,
            "claim_number": claim_number,
            "vin": vin,
            "vehicle": vehicle,
            "score": score
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "⚠️ AI review failed."})
