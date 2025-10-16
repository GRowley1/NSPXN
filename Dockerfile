FROM python:3.11-slim

# System deps for pdf2image (Poppler) — and optional HEIC support
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (merge Presidio into requirements.txt or copy the extra file first)
COPY requirements.txt /app/requirements.txt
# (If you kept a separate file, also: COPY requirements-presidio.txt /app/requirements-presidio.txt)

RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_lg

# If you DID keep a separate presisio file, add:
# RUN pip install --no-cache-dir -r requirements-presidio.txt

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

