FROM python:3

LABEL description="Hello world docker image"

COPY requirements_lock.txt /requirements_lock.txt
RUN pip install -r /requirements_lock.txt
COPY src/entrypoint.py /entrypoint.py

CMD [ "python", "/entrypoint.py" ]
