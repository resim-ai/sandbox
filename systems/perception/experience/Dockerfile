FROM python:3.10-slim

# Build argument to control local dataset copying
ARG LOCAL=true

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
COPY sample_image.png /tmp/sample_image.png
# Install any needed packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source files
COPY owl_model.py .
COPY experience_runner.py .

# Create dataset directory
RUN mkdir -p /dataset

# Copy dataset (will be overridden by volume mounts in production)
COPY --chown=root:root ./dataset /dataset

# Log the mode
RUN if [ "$LOCAL" = "true" ]; then \
        echo "Local Experience Mode"; \
    else \
        echo "Experience expected in S3"; \
    fi

RUN python experience_runner.py --test

# Set entrypoint
ENTRYPOINT ["python", "experience_runner.py"]
