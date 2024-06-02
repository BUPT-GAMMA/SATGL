import os
import sys
import torch

current_dir = os.path.dirname(__file__)
relative_path = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, relative_path)

from satgl.wrappers.data_wrappers.cnf_data_wrappers import CNFDataWrapper
from satgl.config.config import Config
from satgl.data.datasets.cnf_datasets import(
    SatistifiabilityDataset,
    MaxSATDataset,
    UnSATCoreDataset,
    MultiTasksDataset
)
from satgl.data.dataloaders.cnf_dataloaders import(
    SatistifiabilityDataLoader,
    MaxSATDataLoader,
    UnsatCoreDataLoader,
    MultiTasksDataloader
)
from satgl.models.cnf_models.neurosat import NeuroSAT
from satgl.wrappers.model_wrappers.cnf_model_wrappers import MultiTasksCNFModelWrapper
from satgl.evaluator.evaluator import EvaluatorManager
from satgl.models.cnf_models.neurosat import NeuroSAT
from satgl.trainer.trainer import MultiTasksTrainer

def get_trainer(config):
    device = config["device"]
    tasks = config["tasks"]
    early_stopping = config["early_stopping"]
    eval_step = config["eval_step"]
    epochs = config["epochs"]
    save_model = config["save_model"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    log_file = config["log_file"]
    valid_metric = config["valid_metric"]

    return MultiTasksTrainer(
        device=device,
        tasks=tasks,
        early_stopping=early_stopping,
        eval_step=eval_step,
        epochs=epochs,
        save_model=save_model,
        lr=lr,
        weight_decay=weight_decay,
        log_file=log_file,
        valid_metric=valid_metric
    )


if __name__ == "__main__":
    # test settings
    sys.path.insert(0, current_dir)
    current_dir = os.path.dirname(__file__)
    cnf_dir = os.path.join(current_dir, "../benchmarks", "sr", "train")
    label_path = os.path.join(current_dir, "../benchmarks", "sr", "label", "train.csv")
    root_dir = os.path.join(current_dir, "../benchmarks", "sr", )
    config_files = ["./test_yaml/satformer.yaml"]
    parameter_dict = {
        "task": "satisfiability",
        "model": "neurosat",
        "dataset_path": "../benchmarks/sr",
        "tasks": ["satisfiability", "unsat_core"],
        "graph_type": "lcg",
        "batch_size": 32,
        "epochs": 2000
    }
    config = Config(config_files, parameter_dict)

    # # get dataset and dataloader
    # train_dataset = MultiTasksDataset(cnf_dir, label_path, config["graph_type"], config["tasks"])
    # train_dataloader = MultiTasksDataloader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    data = CNFDataWrapper(
        root_dir = root_dir,
        tasks = config["tasks"],
        graph_type = config["graph_type"],
        batch_size = config["batch_size"]
    )

    # trainer
    trainer = get_trainer(config)

    # get model
    model_wrapper = MultiTasksCNFModelWrapper(
        tasks = config["tasks"],
        emb_size = 128,
        graph_type = "lcg",
        num_fc = 3,
        weights = [5, 1],
        losses = [torch.nn.BCELoss(), torch.nn.BCELoss()],
        evaluators = [EvaluatorManager(["accuracy"]), EvaluatorManager(["accuracy"])],
        model = NeuroSAT(config)   
    )

    trainer.train(model_wrapper, data.train_dataloader, data.valid_dataloader, data.test_dataloader)


    

