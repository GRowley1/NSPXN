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

openai.api_key = os.getenv("OPENAI_API_KEY")

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

        if file.content_type.startswith("image/"):
            b64 = base64.b64encode(content).decode("utf-8")
            images.append({
                "type": "image_url",
                "image_url": { "url": f"data:{file.content_type};base64,{b64}" }
            })
        elif name.endswith(".pdf"):
            texts.append(extract_text_from_pdf(io.BytesIO(content)))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(content)))
        elif name.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {file.filename}")

    combined_text = "\n\n".join(texts).strip()
    if not combined_text and not images:
        return JSONResponse(status_code=400, content={"error": "No readable text or images found."})

    vision_input = []
    if combined_text:
        vision_input.append({
            "type": "text",
            "text": f"""Estimate & Client Rules:\n\n{combined_text}\n\nClient Rules:\n{client_rules}"""
        })
    vision_input.extend(images)

    prompt = "You are an AI auto damage auditor. Review the uploaded estimate and photos for accuracy. Flag issues and assess compliance with the provided rules."

    try:
        print("📤 Sending prompt to OpenAI...")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": vision_input}
            ],
            max_tokens=3500
        )

        print("✅ GPT response received.")
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
