FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

# Install minimal system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    wget \
    && apt-get clean

# Install any needed packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source files
COPY *.py .

# Create dataset directory
RUN mkdir -p /dataset

# Copy dataset (build script ensures ./dataset exists)
COPY --chown=root:root ./dataset /dataset

# Dataset is copied from local folder (may be empty if using S3)
RUN echo "Dataset copied to /dataset"

# Set entrypoint
ENTRYPOINT ["python", "entrypoint.py"]