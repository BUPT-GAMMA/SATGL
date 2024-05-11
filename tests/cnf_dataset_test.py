import os
import sys

current_dir = os.path.dirname(__file__)
relative_path = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, relative_path)

from satgl.wrappers.data_wrappers.cnf_data_wrappers import CNFDataWrapper
from satgl.config.config import Config

if __name__ == "__main__":
    tasks = ["satisfiability", "maxsat", "unsat_core"]
    graphs = ["lcg", "vcg"]
    for task in tasks:
        for graph in graphs:
            data_dict = {
                "dataset_path": "./test_benchmarks/test_cnf",
                "graph_type": graph,
                "task": task,
                "log_file": "./test_log/cnf_dataset_test.log"
            }
            print(f"task: {task}, graph: {graph}")
            config = Config(parameter_dict=data_dict)

            batch_size = config["batch_size"]
            root_dir = config["dataset_path"]
            graph_type = config["graph_type"]
            task = config["task"]
            log_file = config["log_file"]
            data = CNFDataWrapper(root_dir, task, graph_type, log_file, batch_size)


