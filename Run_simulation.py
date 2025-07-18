import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
import time
import optuna
import pandas as pd
from datetime import datetime,timedelta
from lifelines.utils import concordance_index

from sklearn.model_selection import KFold, train_test_split

from Data import dataset_positive
from Hyperparameters import read_configuration
from Model import residual_train
from Utils import get_hash, Metrics_RNCQR,to_numpy
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
metric_result_path = 'metric_Our_method.csv'


if __name__ == "__main__":
    run_whole_simulations(do_tuning=True)
