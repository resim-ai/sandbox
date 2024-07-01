FROM python:3

LABEL description="Demo metrics build docker image"

COPY thonk.gif /thonk.gif
COPY plotly.json /plotly.json
COPY metrics.py /metrics.py
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

CMD [ "python", "/metrics.py" ]
