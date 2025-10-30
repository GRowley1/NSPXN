# Slim Python
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for pdf2image (poppler) and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*
  
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*

# App folder
WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
  && python -m spacy download en_core_web_lg

ARG BUILD_ID=1

# Copy app
COPY . /app

# Render listens on $PORT; we’ll default to 10000 like your service
ENV PORT=10000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--proxy-headers"]

