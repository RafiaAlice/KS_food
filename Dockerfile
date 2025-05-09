# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for spaCy, FAISS, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Install the spaCy model the correct way (from pip or URL)
RUN python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# Optionally install Hugging Face CLI (if needed for snapshot_download)
RUN pip install huggingface_hub

# Copy app source code
COPY . .

# Environment setup
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# Expose port for gunicorn
EXPOSE 8080

# Start Flask app with Gunicorn and a generous timeout (set separately in gunicorn_config.py)
CMD ["gunicorn", "-c", "gunicorn_config.py", "--bind", "0.0.0.0:8080", "app:app"]
