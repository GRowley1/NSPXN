FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg
COPY . /app
CMD bash -lc 'uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}'
