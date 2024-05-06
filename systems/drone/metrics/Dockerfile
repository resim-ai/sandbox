FROM python:3.10

LABEL description="Demo metrics build docker image"

COPY requirements_lock.txt /requirements_lock.txt
RUN pip install --no-cache-dir -r /requirements_lock.txt

COPY metrics.py /
COPY decoder_factory.py /
COPY python_libs/ /


ENV PYTHONPATH="/"
CMD [ "python", "/metrics.py" ]
