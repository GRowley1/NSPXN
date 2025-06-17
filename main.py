
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import smtplib
from email.message import EmailMessage
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/send-email")
async def send_email(subject: str = Form(...), body: str = Form(...)):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"

        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)

        return {"status": "Email sent"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
