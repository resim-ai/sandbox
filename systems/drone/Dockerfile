FROM 909785973729.dkr.ecr.us-east-1.amazonaws.com/drone-simulator:swarm-seed

# Install JQ:
RUN apt-get update && apt-get install -y jq

COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
