import torch

from satgl.config.config import Config
from satgl.wrappers.model_wrappers.cnf_model_wrappers import(
    SatisfiabilityModelWrapper,
    MaxSATModelWrapper,
    UnSATCoreModelWrapper
)
from satgl.models.cnf_models.neurosat import NeuroSAT
from satgl.models.cnf_models.gms import GMS
from satgl.models.cnf_models.neurocore import NeuroCore
from satgl.models.cnf_models.nlocalsat import NLocalSAT
from satgl.models.cnf_models.querysat import QuerySAT
from satgl.models.cnf_models.gnn import(
    GCN,
    GIN
)
from satgl.evaluator.evaluator import EvaluatorManager

from typing import Optional, Callable

supported_models = {
    "neurosat": NeuroSAT,
    "gms": GMS,
    "neurocore": NeuroCore,
    "nlocalsat": NLocalSAT,
    "querysat": QuerySAT,
    "gcn": GCN,
    "gin": GIN,
}

supported_tasks = {
    "satisfiability": SatisfiabilityModelWrapper,
    "maxsat": MaxSATModelWrapper,
    "unsat_core": UnSATCoreModelWrapper,
}

tasks_default_loss = {
    "satisfiability": torch.nn.BCELoss(),
    "maxsat": torch.nn.BCELoss(),
    "unsat_core": torch.nn.BCELoss()
}

def get_model(
        config: Config,
        loss_fn: Optional[Callable] = None
    ):
    task = config["task"]
    eval_metrics = config["eval_metrics"]

    # get wrapper for the task
    if task in supported_tasks:
        model_wrapper = supported_tasks[task](config)
    else:
        raise ValueError("Invalid task.")
    
    # set model
    model_name = config["model_settings"]["model"]
    if model_name in supported_models:
        model = supported_models[model_name](config)
    else:
        raise ValueError("Invalid model.")
    model_wrapper.model = model

    # set loss
    if loss_fn is not None:
        model_wrapper.loss = loss_fn
    else:
        model_wrapper.loss = tasks_default_loss[task]
    
    # set evaluator
    model_wrapper.evaluator = EvaluatorManager(eval_metrics)

    return model_wrapper
