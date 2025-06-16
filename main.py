
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI
from fpdf import FPDF
import base64, io, os
from PyPDF2 import PdfReader
from docx import Document
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

LOGO_MAP = {
    "SCA": "SCA-Logo.png",
    "ACD": "ACD-Logo.png",
    "ScoutWorks": "Scout-Works-Logo.png",
    "Sedgwick": "sedgwick-Logo.png"
}

def extract_text_from_pdf(file):
    return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)

def extract_text_from_docx(file):
    return "\n".join(p.text for p in Document(file).paragraphs)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...),
    ia_company: str = Form(...)
):
    images, texts = [], []
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

    vision_message = {"role": "user", "content": []}
    if texts: vision_message["content"].append({"type": "text", "text": "\n\n".join(texts)})
    if images: vision_message["content"].extend(images)

    prompt = f"You are an AI auto damage auditor. Review the uploaded estimate against the damage photos.\nFlag any discrepancies or missing documentation. Confirm compliance with these client rules: {client_rules}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_tokens=3500
        )
        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."

        def extract(label):
            lower = label.lower()
            for line in gpt_output.splitlines():
                if lower in line.lower():
                    return line.split(":")[-1].strip()
            return "N/A"

        metadata = {
            "claim_number": extract("Claim"),
            "vin": extract("VIN"),
            "vehicle": extract("Vehicle"),
            "score": extract("Score") or "N/A",
            "ia_company": ia_company,
            "file_number": file_number,
            "gpt_output": gpt_output
        }

        pdf_path = f"/mnt/data/NSPXN_AI_Review_Report.pdf"
        create_pdf_report(metadata, pdf_path)
        return {"gpt_output": gpt_output, **metadata}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "⚠️ AI review failed."})

def create_pdf_report(data, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    logo_path = f"/mnt/data/{LOGO_MAP.get(data['ia_company'], '')}"
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=150, y=10, w=40)
        pdf.ln(30)
    else:
        pdf.ln(20)

    pdf.cell(0, 10, "NSPXN.com AI Review Report", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%B %d, %Y')}", ln=True)
    pdf.cell(0, 10, f"IA Company: {data['ia_company']}", ln=True)
    pdf.cell(0, 10, f"Claim #: {data['claim_number']}", ln=True)
    pdf.cell(0, 10, f"VIN: {data['vin']}", ln=True)
    pdf.cell(0, 10, f"Vehicle: {data['vehicle']}", ln=True)
    pdf.cell(0, 10, f"Compliance Score: {data['score']}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"""AI Review Summary:
    {data['gpt_output']}""")
    pdf.output(pdf_path)
