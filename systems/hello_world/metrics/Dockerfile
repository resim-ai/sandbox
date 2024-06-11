FROM python:3

LABEL description="Demo metrics build docker image"

COPY thonk.gif /thonk.gif
COPY plotly.json /plotly.json
COPY metrics.py /metrics.py
COPY requirements_lock.txt /requirements_lock.txt
RUN pip install --no-cache-dir -r /requirements_lock.txt

CMD [ "python", "/metrics.py" ]
