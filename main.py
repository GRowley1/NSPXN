
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import io

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

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...)
):
    num_files = len(files)
    print(f"📥 Upload received with {num_files} files, File #: {file_number}")
    print(f"📋 Client rules: {client_rules[:80]}...")

    return {
        "gpt_output": "🧪 Backend is working. This is a static response without GPT processing. "
                      f"{num_files} files received. File #: {file_number}"
    }
