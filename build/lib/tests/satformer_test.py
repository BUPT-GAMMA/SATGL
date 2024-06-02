import os
import sys

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

if __name__ == "__main__":
    # test settings
    current_dir = os.path.dirname(__file__)
    cnf_dir = os.path.join(current_dir, "../benchmarks", "sr", "train")
    label_path = os.path.join(current_dir, "../benchmarks", "sr", "label", "train.csv")

    dataset = MultiTasksDataset(cnf_dir, label_path, "lcg", ["satisfiability", "unsat_core"])
    dataloader = MultiTasksDataloader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        print(batch)

