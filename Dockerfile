# Use an official Python runtime as a parent image
FROM python:3.11-slim

# System dependencies
RUN apt-get update \
    && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \  # Added to provide libGL.so.1
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Set environment variables for headless OpenCV
# Optional: Forces Qt to run without GUI
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV QT_QPA_PLATFORM=offscreen

# Expose port (use $PORT for Render compatibility)
EXPOSE $PORT

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]

