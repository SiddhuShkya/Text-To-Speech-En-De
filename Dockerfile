# Use a lightweight Python 3.12 image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for sentencepiece + audio
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    cmake \
    pkg-config \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]