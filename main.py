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

def extract_field(label, text):
    for line in text.splitlines():
        if label.lower() in line.lower():
            return line.split(":")[-1].strip()
    return "N/A"

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...),
    ia_company: str = Form(None)
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

    prompt = f"""You are an AI auto damage auditor. Review the uploaded estimate against the damage photos.
Flag any discrepancies or missing documentation. Confirm compliance with these client rules: {client_rules}"""

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
        score = extract_field("Score", gpt_output)

        # Save PDF
        logo_map = {
            "SCA": "logos/SCA-Logo.png",
            "ACD": "logos/ACD-Logo.png",
            "ScoutWorks": "logos/Scout-Works-Logo.png",
            "Sedgwick": "logos/sedgwick-Logo.png"
        }
        pdf = FPDF()
        pdf.add_page()
        if ia_company and ia_company in logo_map and os.path.exists(logo_map[ia_company]):
            pdf.image(logo_map[ia_company], x=10, y=8, w=33)
            pdf.ln(25)

        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Claim #: {claim_number}\nVIN: {vin}\nVehicle: {vehicle}\nCompliance Score: {score}\n\nAI Review:\n{gpt_output}")
        os.makedirs("generated", exist_ok=True)
        pdf_path = f"generated/{file_number}.pdf"
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

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"generated/{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"error": "PDF not found"})