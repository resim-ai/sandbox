FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
# Install dependencies (this layer will be cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source files (these change less frequently than dataset)
COPY owl_model.py .
COPY experience_runner.py .
COPY sample_image.png /tmp/sample_image.png

# Create directories
RUN mkdir -p /dataset /experiences

# Copy dataset (this is the largest and most frequently changing layer)
COPY --chown=root:root ./dataset /dataset

# Flatten dataset structure (combine multiple RUN commands)
RUN find /dataset/experience -name "*.png" -exec cp {} /experiences/ \; && \
    find /dataset/experience -name "*-truth.csv" -exec cp {} /experiences/ \; && \
    echo "Dataset copied to /dataset and experience files flattened to /experiences/" && \
    echo "Files in /experiences:" && ls -la /experiences/ | head -10

# Test the runner (this will be cached unless code or dependencies change)
RUN python experience_runner.py --test

# Set entrypoint
ENTRYPOINT ["python", "experience_runner.py"]