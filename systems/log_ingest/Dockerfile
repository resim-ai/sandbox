FROM alpine:latest

LABEL description="Log Ingestion Docker Image"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT "/entrypoint.sh"
