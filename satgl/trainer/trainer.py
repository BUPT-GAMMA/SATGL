import copy
import warnings
from typing import Optional
import numpy as np
from tqdm import tqdm
import os
import shutil

import torch

from dgl.dataloading import GraphDataLoader
from copy import deepcopy
from typing import List

from torch.optim.lr_scheduler import (
    CosineAnnealingLR
)
from satgl.config.config import Config
from satgl.wrappers.model_wrappers.base_model_wrapper import ModelWrapper
from satgl.loggers.logger import Logger
from satgl.evaluator.evaluator import EvaluatorManager

def move_to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [move_to_device(d, device) for d in data]
    if isinstance(data, dict):
        return {key: move_to_device(val, device) for key, val in data.items()}
    return data.to(device)

class Trainer(object):
    def __init__(
        self,
        device: str = "cuda",
        early_stopping: bool = False,
        eval_step: int = 1,
        epochs: int = 100,
        save_model: str = "./save_model",
        lr: float = 1e-3,
        weight_decay: float = 1e-10,
        log_file: bool = False,
        valid_metric: str = "loss"
    ):
        self.device = device
        self.early_stopping = early_stopping
        self.eval_step = eval_step
        self.epochs = epochs
        self.save_model = save_model

        self.lr = lr
        self.weight_decay = weight_decay    
        self.log_file = log_file

        self.valid_metric = valid_metric

        if not os.path.exists(save_model):
            os.makedirs(save_model)

        self.logger = Logger(log_file)

    def set_scheduler(self, optimizer: torch.optim.Optimizer):
        self.scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

    def set_optimizer(self, model: ModelWrapper):
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay)
        )

    def train_epoch(
            self,
            model: ModelWrapper,
            data_loader: GraphDataLoader
            ) -> dict:
        r"""
        Train the model for one epoch.

        Parameters
        ----------
        model : ModelWrapper
            The model to train.
        train_loader : GraphDataLoader
            The train data loader.
        """
        model.train()
        model.evaluator.reset()
        sum_loss = 0
        data_size = 0

        iter_data = (
            tqdm(
                data_loader,
                total=len(data_loader),
                ncols=100,
                desc=f"train "
            )
        )

        for batch in iter_data:
            self.optimizer.zero_grad()
            model.to(self.device)
            batch = move_to_device(batch, self.device)
            loss, results = model.train_step(batch)
            loss.backward()
            self.optimizer.step()

            batch_data_size = len(batch["label"])
            data_size += batch_data_size
            sum_loss += loss.item() * batch_data_size

            iter_data.set_postfix({"loss": f"{sum_loss / data_size:.4f}"})
        
        return sum_loss / data_size

    def evaluate(
            self,
            model: ModelWrapper,
            data_loader: GraphDataLoader,
            ) -> dict:
        r"""
        Evaluate the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        valid_loader : GraphDataLoader
            The validation data loader.

        Returns
        -------
        dict
            The evaluation results.
        """
        model.eval()
        model.evaluator.reset()
        sum_loss = 0
        data_size = 0

        iter_data = (
            tqdm(
                data_loader,
                total=len(data_loader),
                ncols=100,
                desc=f"eval "
            )
        )


        for batch in iter_data:
            batch = move_to_device(batch, self.device)
            loss, results = model.valid_step(batch)

            batch_data_size = len(batch["label"])
            data_size += batch_data_size
            sum_loss += loss.item() * batch_data_size

            iter_data.set_postfix({"loss": f"{sum_loss / data_size:.4f}"})

        return sum_loss / data_size
    
    def train(
            self, 
            model: ModelWrapper,
            train_loader: GraphDataLoader,
            valid_loader: GraphDataLoader = None,
            test_loader: GraphDataLoader = None,
            use_best_model: bool = True
            ) -> dict:
        r"""
        Train the model on the given dataset.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to train.
        train_loader : GraphDataLoader
            The train data loader.
        valid_loader : GraphDataLoader
            The validation data loader.
        test_loader : GraphDataLoader
            The test data loader.   
        use_best_model : bool
            Whether to use the best model during evaluation.
        """
        self.set_optimizer(model)
        best_evaluator = deepcopy(model.evaluator)
        best_evaluator.reset()
        best_epoch = None
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"train: [{epoch}/{self.epochs}]")
            train_loss = self.train_epoch(model, train_loader)

            # log train and valid results
            self.logger.display_eval_results(model.evaluator.get_eval_results())
            if self.eval_step and epoch % self.eval_step == 0 and valid_loader is not None:
                valid_loss = self.evaluate(model, valid_loader)
                self.logger.info(f"valid: [{epoch}/{self.epochs}]")
                self.logger.display_eval_results(model.evaluator.get_eval_results())
            
                # update best results
                if self.valid_metric == "loss":
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        best_epoch = epoch
                        best_evaluator = deepcopy(model.evaluator)
                else:
                    cur_results = model.evaluator[self.valid_metric]
                    best_results = best_evaluator[self.valid_metric]
                    if cur_results.better(best_results):
                        best_evaluator = deepcopy(model.evaluator)
                        best_epoch = epoch

                save_model_file = os.path.join(self.save_model, f"{epoch}_model.pth")
                torch.save(model.state_dict(), save_model_file)
        
        best_model_file = os.path.join(self.save_model, f"{best_epoch}_model.pth")
        best_model_save_file = os.path.join(self.save_model, "best_model.pth")
        shutil.copyfile(best_model_file, best_model_save_file)

        if test_loader is not None:
            if use_best_model:
                model.load_state_dict(torch.load(best_model_file))
            
            self.logger.info(f"best epoch: {best_epoch}")
            self.logger.info(f"best model file: {best_model_file}")
            self.logger.info(f"test results: ")
            self.evaluate(model, test_loader)
            eval_results = model.evaluator.get_eval_results()
            self.logger.display_eval_results(eval_results)


class MultiTasksTrainer(object):
    def __init__(
        self,
        tasks: List[str],
        device: str = "cuda",
        early_stopping: bool = False,
        eval_step: int = 1,
        epochs: int = 100,
        save_model: str = "./save_model",
        lr: float = 1e-3,
        weight_decay: float = 1e-10,
        log_file: bool = False,
        valid_metric: str = "loss"
    ):
        self.device = device
        self.early_stopping = early_stopping
        self.eval_step = eval_step
        self.epochs = epochs
        self.save_model = save_model

        self.lr = lr
        self.weight_decay = weight_decay    
        self.log_file = log_file

        self.tasks = tasks
        self.valid_metric = valid_metric

        if not os.path.exists(save_model):
            os.makedirs(save_model)

        self.logger = Logger(log_file)

    def set_scheduler(self, optimizer: torch.optim.Optimizer):
        self.scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

    def set_optimizer(self, model: ModelWrapper):
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay)
        )

    def train_epoch(
            self,
            model: ModelWrapper,
            data_loader: GraphDataLoader
            ) -> dict:
        r"""
        Train the model for one epoch.

        Parameters
        ----------
        model : ModelWrapper
            The model to train.
        train_loader : GraphDataLoader
            The train data loader.
        """
        model.train()
        for evaluator in model.evaluators:
            evaluator.reset()
        sum_loss = 0
        data_size = 0

        iter_data = (
            tqdm(
                data_loader,
                total=len(data_loader),
                ncols=100,
                desc=f"train "
            )
        )

        for batch in iter_data:
            self.optimizer.zero_grad()
            model.to(self.device)
            batch = move_to_device(batch, self.device)
            loss, results = model.train_step(batch)
            loss.backward()
            self.optimizer.step()

            batch_data_size = len(batch["label"])
            data_size += batch_data_size
            sum_loss += loss.item() * batch_data_size

            iter_data.set_postfix({"loss": f"{sum_loss / data_size:.4f}"})
        
        return sum_loss / data_size

    def evaluate(
            self,
            model: ModelWrapper,
            data_loader: GraphDataLoader,
            ) -> dict:
        r"""
        Evaluate the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        valid_loader : GraphDataLoader
            The validation data loader.

        Returns
        -------
        dict
            The evaluation results.
        """
        model.eval()
        for evaluator in model.evaluators:
            evaluator.reset()
        sum_loss = 0
        data_size = 0

        iter_data = (
            tqdm(
                data_loader,
                total=len(data_loader),
                ncols=100,
                desc=f"eval "
            )
        )


        for batch in iter_data:
            batch = move_to_device(batch, self.device)
            loss, results = model.valid_step(batch)

            batch_data_size = len(batch["label"])
            data_size += batch_data_size
            sum_loss += loss.item() * batch_data_size

            iter_data.set_postfix({"loss": f"{sum_loss / data_size:.4f}"})

        return sum_loss / data_size
    
    def train(
            self, 
            model: ModelWrapper,
            train_loader: GraphDataLoader,
            valid_loader: GraphDataLoader = None,
            test_loader: GraphDataLoader = None,
            use_best_model: bool = True
            ) -> dict:
        r"""
        Train the model on the given dataset.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to train.
        train_loader : GraphDataLoader
            The train data loader.
        valid_loader : GraphDataLoader
            The validation data loader.
        test_loader : GraphDataLoader
            The test data loader.   
        use_best_model : bool
            Whether to use the best model during evaluation.
        """
        self.set_optimizer(model)
        best_evaluator = deepcopy(model.evaluators[0])
        best_evaluator.reset()
        best_epoch = None
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"train: [{epoch}/{self.epochs}]")
            train_loss = self.train_epoch(model, train_loader)

            # log train and valid results
            train_results = {}
            for idx, task in enumerate(self.tasks):
                for key, value in model.evaluators[idx].get_eval_results().items():
                    train_results[task + "_" + key] = value
            self.logger.display_eval_results(train_results)
            if self.eval_step and epoch % self.eval_step == 0 and valid_loader is not None:
                valid_loss = self.evaluate(model, valid_loader)
                self.logger.info(f"valid: [{epoch}/{self.epochs}]")
                valid_results = {}
                for idx, task in enumerate(self.tasks):
                    for key, value in model.evaluators[idx].get_eval_results().items():
                        valid_results[task + "_" + key] = value
                self.logger.display_eval_results(valid_results)
            
                # update best results
                if self.valid_metric == "loss":
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        best_epoch = epoch
                        best_evaluator = deepcopy(model.evaluators[0])
                else:
                    cur_results = model.evaluators[0][self.valid_metric]
                    best_results = best_evaluator[self.valid_metric]
                    if cur_results.better(best_results):
                        best_evaluator = deepcopy(model.evaluators[0])
                        best_epoch = epoch

                save_model_file = os.path.join(self.save_model, f"{epoch}_model.pth")
                torch.save(model.state_dict(), save_model_file)
        
        best_model_file = os.path.join(self.save_model, f"{best_epoch}_model.pth")
        best_model_save_file = os.path.join(self.save_model, "best_model.pth")
        shutil.copyfile(best_model_file, best_model_save_file)

        if test_loader is not None:
            if use_best_model:
                model.load_state_dict(torch.load(best_model_file))
            
            self.logger.info(f"best epoch: {best_epoch}")
            self.logger.info(f"best model file: {best_model_file}")
            self.logger.info(f"test results: ")
            self.evaluate(model, test_loader)
            eval_results = model.evaluators[0].get_eval_results()
            self.logger.display_eval_results(eval_results)


            
            
            
            
            
            
        




