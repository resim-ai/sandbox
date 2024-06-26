

import json
from dataclasses import dataclass
import copy

@dataclass
class ExperienceConfig:
    arguments: list[float]
    num_iterations: int

EXPERIENCE_PATH = "/tmp/resim/inputs/experience.json"

def load_experience() -> ExperienceConfig:
    with open(EXPERIENCE_PATH, "r") as f:
        config_json = json.load(f)
    return ExperienceConfig(**config_json)

def compute_sine(args: float, iterations: int):
    log = []

    sums = [0.0] * len(args)
    for i in range(iterations):
        degree = 2*i + 1
        sign =  1. if (i % 2 == 0) else  -1.
        terms = [sign * arg**degree for arg in args]
        for j in range(2, degree + 1):
            terms = [term / j for term in terms]
        for ii, t in enumerate(terms):
            sums[ii] += t

        NS_PER_STEP = 1000000000
        log.append({ "iteration": i, "partial_sums": copy.deepcopy(sums), "time": NS_PER_STEP * i})

    with open("/tmp/resim/outputs/log.json", "w") as l:
        json.dump(log, l)

def main():
    config: ExperienceConfig = load_experience()
    print("Hello, world!")
    compute_sine(config.arguments, config.num_iterations)

if __name__ == "__main__":
    main()
