FROM ubuntu:jammy

RUN head -c 10G </dev/urandom >/bigfile

RUN apt update && apt -y install build-essential python3 python3-pip

LABEL description="Load test build docker image"

COPY /src/ /src
COPY requirements_lock.txt /requirements_lock.txt
RUN pip install --no-cache-dir -r /requirements_lock.txt

RUN gcc -o memory_load src/memory_load.c

CMD [ "python3", "/src/entrypoint.py" ]
