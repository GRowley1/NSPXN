# Use an official Python runtime as a parent image 
FROM python:3.11-slim

# System dependencies
RUN apt-get update \
    && apt-get install -y tesseract-ocr poppler-utils libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Set environment variables
ENV PYTHONIOENCODING=UTF-8
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV QT_QPA_PLATFORM=offscreen

# Render requires binding to this dynamic port
ENV PORT=10000
EXPOSE 10000

# Start the app with dynamic port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

