import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(__file__)
relative_path = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, relative_path)

from satgl.run_experiment import run_experiment
from satgl.config.config import Config

if __name__ == "__main__":
    config_files = ["./test_yaml/gin.yaml"]
    parameter_dict = {
        "task": "maxsat",
        "model": "gin",
        "dataset_path": "../benchmarks/sr",
        "graph_type": "lcg"
    }
    config = Config(config_files, parameter_dict)
    run_experiment(config)