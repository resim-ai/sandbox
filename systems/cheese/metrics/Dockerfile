FROM python:3

LABEL description="Cheese metrics build docker image"

COPY cheese_metrics.py /cheese_metrics.py
COPY requirements_lock.txt /requirements_lock.txt
RUN pip install --no-cache-dir -r /requirements_lock.txt

CMD [ "python", "/cheese_metrics.py" ]
