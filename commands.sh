# Projects Creation
./resim projects list
./resim projects create --name "Perception Metrics 2.0 PoC" --description "Perception Metrics 2.0 PoC"
./resim projects select  "Perception Metrics 2.0 PoC"

# Systems Creation:
./resim systems create --name "perception" --description "A self driving car's perception system"

# Experiences:
./resim experiences create --name "Drone Sandbox Experience 1" --description "Drone Sandbox Experience 1" --location "s3://rerun-staging-experiences/sandbox/01431a13-0f4b-43ca-b6fb-c8c610a66bcf/"
./resim experiences create --name "Drone Sandbox Experience 2" --description "Drone Sandbox Experience 2" --location "s3://rerun-staging-experiences/sandbox/0159a03d-224a-45db-b5e3-63bff7cd4fa3/"
./resim experiences create --name "Drone Sandbox Experience 3" --description "Drone Sandbox Experience 3" --location "s3://rerun-staging-experiences/sandbox/01f4ace1-b8b7-485b-a672-19c6f005c4c5/"


#### build and push the image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 909785973729.dkr.ecr.us-east-1.amazonaws.com
docker build -t perception_test .


docker tag perception_test:latest 909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:perception_test_lain
docker push 909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:perception_test_lain

# A build:
./resim builds create --name "Perception System Build" --branch "main" --version "1.0.0"  --auto-create-branch \
 --description "A build for the Perception System" --system "perception" \
 --image "909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:perception_test_lain"

# # Creating the new metrics build, revising etc.
#  ./resim metrics-builds create --name "Drone Sandbox Metrics 2.0 Build" --systems "Drone Sandbox" --image "909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:drone_sim_metrics2" --version "1.0.0" 
#  #metrics_build_id=583579bd-800f-41f4-87ad-39ec42497dae
 
 # Sync 
 ./resim metrics sync

 ./resim test-suites create --name "Drone Sandbox M2" --description "A test suite for the Drone Sandbox" --system "Drone Sandbox" --metrics-build "583579bd-800f-41f4-87ad-39ec42497dae" --metrics-set "first" --experiences "Drone Sandbox Experience 1","Drone Sandbox Experience 2","Drone Sandbox Experience 3"
 ./resim test-suites run --test-suite "Drone Sandbox M2" --build-id "dd06030f-87c9-4498-a16e-bfab993e5077" --batch-name "Metrics 2 No Set" --pool-labels "resim:metrics2:k8s"


-----
./resim test-suites run --test-suite "truck_follow_safety_set" --build-id "4a117ffc-a864-41a4-830b-aa32c072f4c7"

./resim test-suites run --test-suite "truck_follow_safety_set" --build-id "4a117ffc-a864-41a4-830b-aa32c072f4c7" --sync-metrics-config --pool-labels "resim:metrics2:k8s"