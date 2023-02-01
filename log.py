import wandb


def start_log(model):
    wandb.init(project="Smile-Detector")
    wandb.watch(model, log_freq=75)

def log_epoch(data):
    wandb.log(data)

def stop_log():
    wandb.finish()