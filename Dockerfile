# Use a slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git build-essential curl python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model properly via pip
RUN python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# Optionally install Hugging Face CLI
RUN pip install huggingface_hub

# Copy the application code
COPY . .

# Environment variables
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# Expose the app port
EXPOSE 8080

# Start Flask with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
