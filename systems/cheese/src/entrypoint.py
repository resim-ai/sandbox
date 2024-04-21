

import json
from pathlib import Path
import random 

def read_cheese_file():
  # Default file location
  input_dir = Path("/tmp/resim/inputs")
  cheese_file = input_dir.joinpath("cheese.json")
  cheese = "no cheese :("
  with cheese_file.open() as f:
    cheese_data = json.loads(f.read()) 
    cheese = cheese_data['cheese']
  return cheese

def print_cheese(cheese):
  print("Hello World! My pizza is topped with {}.".format(cheese))

def log_cheese_score(cheese):
  score = random.uniform(0.71, 0.99)
  if cheese == "parmesan":
    score = random.uniform(0.59, 0.70)
  elif cheese == "buratta":
    score = random.uniform(0.3, 0.58)
  elif cheese == "vegan cheeze":
    score = 1.0

  log_data = {"score": score}
  
  output_dir = Path("/tmp/resim/outputs")
  log_file = output_dir.joinpath("cheese_log.json")
  with log_file.open("w") as f:
    json.dump(log_data, f)
     

print_cheese(read_cheese_file())
log_cheese_score(read_cheese_file())
