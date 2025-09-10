FROM python:3.10-slim

# Build argument to control local dataset copying
ARG LOCAL=true

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

# Copy dataset (will be overridden by volume mounts in production)
COPY --chown=root:root ./dataset /dataset

# Log the mode
RUN if [ "$LOCAL" = "true" ]; then \
        echo "Local Metrics Mode"; \
    else \
        echo "Metrics expected in S3"; \
    fi

# Set entrypoint
ENTRYPOINT ["python", "entrypoint.py"]