import os
import torch
import torch.nn as nn

from typing import Union, List

from satgl.loggers.logger import Logger
from satgl.evaluator.evaluator import EvaluatorManager


class ModelWrapper(nn.Module):
    def __init__(self):
        super(ModelWrapper, self).__init__()
        self._loss = None
        self._evaluator = None
        self._model = None

    def calculate_loss(self, batch: dict) -> torch.Tensor:
        output = batch["output"]
        label = batch["label"].float()
        loss = self.loss(output, label)
        return loss
    
    def calculate_eval_metric(self, batch: dict) -> dict:
        output = batch["output"]
        label = batch["label"].float()
        self.evaluator.update_evaluators(output, label)
        results = self.evaluator.get_eval_results()
        return results
    
    def train_step(self, batch: dict):
        batch_out = self.forward(batch)
        loss = self.calculate_loss(batch)
        results = self.calculate_eval_metric(batch)
        del batch_out["output"]
        return loss, results
    
    def valid_step(self, batch: dict):
        batch_out = self.forward(batch)
        loss = self.calculate_loss(batch)
        results = self.calculate_eval_metric(batch)
        del batch_out["output"]
        return loss, results

    def eval_step(self, batch: dict):
        batch_out = self.forward(batch)
        results = self.calculate_eval_metric(batch)
        del batch_out["output"]
        return results

    def pre_stage(self, batch: dict):
        raise NotImplementedError

    def post_stage(self, batch: dict):
        raise NotImplementedError
    
    def forward(self, batch: dict):
        raise NotImplementedError
    
    @property
    def loss(self):
        return self._loss
    
    @property
    def evaluator(self):
        return self._evaluator

    @property
    def model(self):
        return self._model   
    
    @model.setter
    def model(self, model: nn.Module):
        self._model = model 
    
    @loss.setter
    def loss(self, loss):
        self._loss = loss
    
    @evaluator.setter
    def evaluator(self, evaluator: EvaluatorManager):
        self._evaluator = evaluator

class MultitasksModelWrapper(nn.Module):
    def __init__(
            self, 
            tasks: List[str], 
            weights: List[float],
            losses: List,
            evaluators: List[EvaluatorManager],
            model: nn.Module
        ):
        super(MultitasksModelWrapper, self).__init__()
        self.tasks = tasks
        self.weights = weights
        self.losses = losses
        self.evaluators = evaluators
        self.model = model

    def calculate_loss(self, batch: dict) -> torch.Tensor:
        all_loss = 0
        sum_weight = sum(self.weights)
        for idx, task in enumerate(self.tasks):
            output = batch["output"][idx]
            label = batch["label"][idx].float()
            loss = self.losses[idx](output, label)
            all_loss += self.weights[idx] * loss
        all_loss /= sum_weight
        return all_loss
    
    def calculate_eval_metric(self, batch: dict) -> dict:
        all_results = {}
        for idx, task in enumerate(self.tasks):
            output = batch["output"][idx]
            label = batch["label"][idx].float()
            self.evaluators[idx].update_evaluators(output, label)
            results = self.evaluators[idx].get_eval_results()
            for key, value in results.items():
                all_results[task + "_" + key] = value
        return all_results
    
    def train_step(self, batch: dict):
        batch_out = self.forward(batch)
        loss = self.calculate_loss(batch)
        results = self.calculate_eval_metric(batch)
        del batch_out["output"]
        return loss, results
    
    def valid_step(self, batch: dict):
        batch_out = self.forward(batch)
        loss = self.calculate_loss(batch)
        results = self.calculate_eval_metric(batch)
        del batch_out["output"]
        return loss, results

    def eval_step(self, batch: dict):
        batch_out = self.forward(batch)
        results = self.calculate_eval_metric(batch)
        del batch_out["output"]
        return results

    def pre_stage(self, batch: dict):
        raise NotImplementedError

    def post_stage(self, batch: dict):
        raise NotImplementedError
    
    def forward(self, batch: dict):
        raise NotImplementedError



    