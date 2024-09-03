FROM python:3.10-slim

LABEL description="Demo metrics build docker image"

COPY requirements_lock.txt /requirements_lock.txt
RUN pip install --no-cache-dir -r /requirements_lock.txt

COPY metrics.py /
COPY batch_metrics.py /
COPY decoder_factory.py /

CMD [ "python", "/metrics.py" ]
