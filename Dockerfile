# Use official slim Python image
FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip show uvicorn || { echo "uvicorn installation failed"; exit 1; }

# Copy font file for PDF generation
COPY DejaVuSans.ttf /app/DejaVuSans.ttf

# Copy all source files
COPY . .

# Set env vars
ENV PYTHONIOENCODING=UTF-8
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV QT_QPA_PLATFORM=offscreen

# Expose dynamic Render port
EXPOSE $PORT

# Start FastAPI via uvicorn (use $PORT)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
