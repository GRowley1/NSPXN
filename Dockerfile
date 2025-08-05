FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    openai \
    python-docx \
    fpdf \
    pytesseract \
    pdf2image \
    pillow \
    smtplib \
    imagehash \
    ultralytics

ENV OPENAI_API_KEY=your-openai-api-key-here

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
