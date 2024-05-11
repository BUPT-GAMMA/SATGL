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
    config_files = ["./test_yaml/querysat.yaml"]
    parameter_dict = {
        "task": "satisfiability",
        "model": "querysat",
        "dataset_path": "../benchmarks/sr",
    }
    config = Config(config_files, parameter_dict)
    run_experiment(config)