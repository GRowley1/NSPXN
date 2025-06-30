# Use an official Python runtime as a base image
FROM python:3.12-slim

# Install system dependencies needed for OCR
RUN apt-get update && \
    apt-get install -y tesseract-ocr poppler-utils libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Render
ENV PORT=10000
EXPOSE 10000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
