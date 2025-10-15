FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_lg

COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

