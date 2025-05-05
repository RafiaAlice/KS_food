# Use a slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and git-lfs
RUN apt-get update && apt-get install -y git git-lfs build-essential && \
    git lfs install

# Install Python dependencies (from requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download TinyLlama model locally (instead of loading from HF every time)
RUN git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 /models/tinyllama

# Copy application code after model is installed
COPY . .

# Optional: set huggingface cache to prevent re-downloading other models
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# Expose the default port
EXPOSE 8080

# Start the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
