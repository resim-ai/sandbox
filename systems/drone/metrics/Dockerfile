FROM python:3.10-slim

LABEL description="Demo metrics build docker image"

COPY resim_open_core-0.0.0-py3-none-manylinux2014_x86_64.whl /resim_open_core-0.0.0-py3-none-manylinux2014_x86_64.whl
COPY requirements_lock.txt /requirements_lock.txt
RUN pip install --no-cache-dir -r /requirements_lock.txt

COPY metrics.py /
COPY batch_metrics.py /
COPY decoder_factory.py /

CMD [ "python", "/metrics.py" ]
