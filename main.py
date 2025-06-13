
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import openai
import base64
import io
import os
from PyPDF2 import PdfReader
from docx import Document

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
    file_number: str = Form(...)
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
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                vision_message
            ],
            max_tokens=1500
        )

        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."

        def extract_between(label):
            lower = label.lower()
            for line in gpt_output.splitlines():
                if lower in line.lower():
                    return line.split(":")[-1].strip()
            return "N/A"

        return {
            "gpt_output": gpt_output,
            "claim_number": extract_between("Claim"),
            "vin": extract_between("VIN"),
            "vehicle": extract_between("Vehicle"),
            "score": extract_between("Score") or "N/A"
        }

    except Exception as e:
        print("❌ GPT Error:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "gpt_output": "⚠️ AI review failed."}
        )
