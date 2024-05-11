import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(__file__)
relative_path = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, relative_path)

from satgl.wrappers.data_wrappers.cnf_data_wrappers import CNFDataWrapper
from satgl.wrappers.model_wrappers.base_model_wrapper import ModelWrapper
from satgl.wrappers.model_wrappers.cnf_model_wrappers import(
    SatisfiabilityModelWrapper,
    MaxSATModelWrapper,
    UnSATCoreModelWrapper
)
from satgl.config.config import Config
from satgl.models.cnf_models.neurosat import NeuroSAT
from satgl.evaluator.evaluator import EvaluatorManager

if __name__ == "__main__":
    # data wrapper tests
    tasks = ["satisfiability", "maxsat", "unsat_core"]
    graphs = ["hlcg", "hvcg", "lcg", "vcg", "lig", "vig"]
    for task in tasks:
        for graph in graphs:
            data_dict = {
                "dataset_path": "./test_benchmarks/test_cnf",
                "graph": graph,
                "task": task
            }
            print(f"task: {task}, graph: {graph}")
            config = Config(parameter_dict=data_dict)
            data = CNFDataWrapper(config) 


    # satisfiability model wrapper tests
    data_dict = {
        "dataset_path": "./test_benchmarks/test_cnf",
        "graph": "lcg",
        "task": "satisfiability",
        "eval_metrics": ["accuracy"]
    }
    config_files = ["./test_yaml/neurosat.yaml"]
    config = Config(config_files, parameter_dict=data_dict)
    data = CNFDataWrapper(config)
    model = SatisfiabilityModelWrapper(config)
    model.model = NeuroSAT(config)
    model.loss = torch.binary_cross_entropy_with_logits
    model.evaluator = EvaluatorManager(config)
    
    for batch in data.train_dataloader:
        model.train_step(batch)

    # maxsat model wrapper tests
    data_dict = {
        "dataset_path": "./test_benchmarks/test_cnf",
        "graph": "lcg",
        "task": "maxsat"
    }
    config_files = ["./test_yaml/neurosat.yaml"]
    config = Config(config_files, parameter_dict=data_dict)
    data = CNFDataWrapper(config)
    model = MaxSATModelWrapper(config)
    model.model = NeuroSAT(config)
    model.loss = torch.binary_cross_entropy_with_logits
    model.evaluator = EvaluatorManager(config)


    
    for batch in data.train_dataloader:
        model.train_step(batch)


    # unsat core model wrapper tests
    data_dict = {
        "dataset_path": "./test_benchmarks/test_cnf",
        "graph": "lcg",
        "task": "unsat_core"
    }
    config_files = ["./test_yaml/neurosat.yaml"]
    config = Config(config_files, parameter_dict=data_dict)
    data = CNFDataWrapper(config)
    model = UnSATCoreModelWrapper(config)
    model.model = NeuroSAT(config)
    model.loss = torch.binary_cross_entropy_with_logits
    model.evaluator = EvaluatorManager(config)


    for batch in data.train_dataloader:
        model.train_step(batch)


