FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
COPY sample_image.png /tmp/sample_image.png
# Install any needed packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source files
COPY owl_model.py .
COPY experience_runner.py .

RUN python experience_runner.py --test

# Set entrypoint
ENTRYPOINT ["python", "model_analyzer.py","--test"]
