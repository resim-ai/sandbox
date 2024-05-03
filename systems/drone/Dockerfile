FROM ubuntu:jammy

RUN apt update && apt -y upgrade
RUN apt -y install uuid-dev

ARG RESIM_RUN_RUNFILES_DIR

COPY bin/ /resim/
COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
