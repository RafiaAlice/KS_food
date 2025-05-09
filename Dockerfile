# Use a slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face CLI to support snapshot_download
RUN pip install huggingface_hub

# Copy your application code
COPY . .

# Environment setup
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# Expose the default port
EXPOSE 8080

# Start the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
