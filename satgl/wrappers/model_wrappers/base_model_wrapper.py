import os
import torch
import torch.nn as nn

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




    