FROM ubuntu:jammy

RUN apt update && apt -y upgrade
RUN apt -y install uuid-dev python3 python3-pip

ARG RESIM_RUN_RUNFILES_DIR

COPY bin/ /resim/
COPY entrypoint.sh /
COPY apply_params.py /

RUN pip3 install protobuf

ENTRYPOINT ["/entrypoint.sh"]
