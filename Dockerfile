# 1️⃣ Use a slim official Python base image
FROM python:3.12-slim

# 2️⃣ Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy your app files
COPY . /app
WORKDIR /app

# 5️⃣ Expose the port
EXPOSE 10000

# 6️⃣ Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
