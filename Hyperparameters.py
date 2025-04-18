
import json

import optuna
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from functools import partial





def read_configuration(config_file):
    with open(config_file,"r") as file:
        config = json.load(file)
    return config

def get_hyperparameters(config, scenario):
    return config[scenario]




