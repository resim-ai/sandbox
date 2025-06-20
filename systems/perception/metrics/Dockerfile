FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

# Install any needed packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source files
COPY *.py .

# Set entrypoint
ENTRYPOINT ["python", "entrypoint.py"]