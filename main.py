
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
    print("🔐 OpenAI Key starts with:", os.getenv("OPENAI_API_KEY")[:10])
    images = []
    texts = []

    for file in files:
        content = await file.read()
        name = file.filename.lower()
        print(f"📄 Processing file: {name}")

        if name.endswith((".jpg", ".jpeg", ".png")):
            b64 = base64.b64encode(content).decode("utf-8")
            images.append({
                "type": "image_url",
                "image_url": { "url": f"data:image/jpeg;base64,{b64}" }
            })
        elif name.endswith(".pdf"):
            extracted = extract_text_from_pdf(io.BytesIO(content))
            texts.append(extracted)
        elif name.endswith(".docx"):
            extracted = extract_text_from_docx(io.BytesIO(content))
            texts.append(extracted)
        elif name.endswith(".txt"):
            decoded = content.decode("utf-8", errors="ignore")
            texts.append(decoded)
        else:
            skipped = f"⚠️ Skipped unsupported file: {file.filename}"
            texts.append(skipped)
            print(skipped)

    print("📥 Files received - Texts:", len(texts), "Images:", len(images))

    prompt = f"""You are an AI auto damage auditor. Compare the following estimate text with the uploaded vehicle damage photos. 
Note any discrepancies between visible damage and written line items. Then verify if the estimate follows client rules:
{client_rules}"""

    messages = [{"role": "system", "content": prompt}]
    if texts:
        messages.append({ "role": "user", "content": "\n\n".join(texts) })
    if images:
        messages.append({ "role": "user", "content": images })

    print("🧠 Sending prompt to GPT...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500
        )
        print("✅ GPT response received.")
        output = response.choices[0].message.content
        if not output:
            output = "⚠️ GPT returned an empty response."
        return { "gpt_output": output }
    except Exception as e:
        print("❌ GPT Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "⚠️ AI review failed. Please try again."})
