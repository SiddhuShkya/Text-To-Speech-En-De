# Use CUDA runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-distutils \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "main.py"]
