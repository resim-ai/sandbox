./metrics/build.sh 
docker tag hello_world_metrics:latest 909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:hello_world_metrics_lain
docker push  909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:hello_world_metrics_lain