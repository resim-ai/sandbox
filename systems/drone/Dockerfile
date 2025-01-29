FROM 909785973729.dkr.ecr.us-east-1.amazonaws.com/drone-simulator:swarm-seed

ARG VELOCITY_COST_OVERRIDE
ENV VELOCITY_COST_OVERRIDE=$VELOCITY_COST_OVERRIDE

COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
