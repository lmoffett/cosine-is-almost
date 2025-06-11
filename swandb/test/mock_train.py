import argparse
import time

import wandb


def train(param1: int = 1, param2: int = 10, baseline: int = 1000, sleep: int = 0):
    """
    Simulate model training with hyperparameters and log results to Weights & Biases.

    Args:
        param1 (int): First hyperparameter
        param2 (int): Second hyperparameter
    """

    print(f"Sleeping for {sleep} seconds")
    time.sleep(sleep)
    # Calculate target metric (simple sum of parameters)
    target_metric = param1 + param2 + baseline

    # Log the metric
    wandb.log({"mock_target_metric": target_metric})

    return target_metric


if __name__ == "__main__":

    wandb.init(project="test")

    parser = argparse.ArgumentParser(description="Hyperparameter sweep demo")

    parser.add_argument("--param1", type=int, default=1, help="First hyperparameter")

    parser.add_argument("--param2", type=int, default=10, help="Second hyperparameter")

    parser.add_argument(
        "--baseline", type=int, default=1000, help="hyper-sweep-parameter"
    )

    parser.add_argument(
        "--sleep", type=int, default=0, help="pause for a number of seconds"
    )

    args = parser.parse_args()

    # Run training with parsed arguments
    result = train(
        param1=args.param1, param2=args.param2, baseline=args.baseline, sleep=args.sleep
    )

    # End the wandb run
    wandb.finish()
