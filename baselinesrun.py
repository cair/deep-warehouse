import sys
from baselines.baselines import run
if __name__ == "__main__":

    run.main({
        "--alg": "ppo2"
    })