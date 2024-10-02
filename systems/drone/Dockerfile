FROM 909785973729.dkr.ecr.us-east-1.amazonaws.com/drone-simulator:0de4571c4816b18d3f2d917a3be84d7d3b7fb182

# Install JQ:
RUN apt-get update && apt-get install -y jq

COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
