FROM python:3.10-slim

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

# Copy dataset (build script ensures ./dataset exists)
COPY --chown=root:root ./dataset /dataset

# Dataset is copied from local folder (may be empty if using S3)
RUN echo "Dataset copied to /dataset"

RUN python experience_runner.py --test

# Set entrypoint
ENTRYPOINT ["python", "experience_runner.py"]
