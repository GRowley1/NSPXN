
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import openai
import base64
import io
from PyPDF2 import PdfReader
from docx import Document

app = FastAPI()

# Enable CORS
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

# Utility to extract text from PDFs
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in pdf.pages)

# Utility to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

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

    prompt = f"""You are an AI auditor. Compare these estimate documents to the uploaded vehicle damage photos. 
Highlight any mismatches between visual damage and written repair line items. 
Then check compliance with the following client rules: {client_rules}"""

    messages = [{"role": "system", "content": prompt}]
    if texts:
        messages.append({ "role": "user", "content": "\n\n".join(texts) })
    if images:
        messages.append({ "role": "user", "content": images })

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500
        )
        return { "gpt_output": response.choices[0].message.content }
    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })

@app.get("/")
async def root():
    return { "status": "ok" }
