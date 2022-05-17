from train import _train, wandb
from config import get_sweep_config

if __name__ == "__main__":
    sweep_id = wandb.sweep(get_sweep_config())
    count = 5
    wandb.agent(sweep_id, function=_train, count=count)