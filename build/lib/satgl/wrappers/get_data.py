from satgl.wrappers.data_wrappers.cnf_data_wrappers import CNFDataWrapper


cnf_tasks = ["satisfiability", "maxsat", "unsat_core"]

def get_data(config):
    root_dir = config["dataset_path"]
    task = config["task"]
    graph_type = config["graph_type"]
    log_file = config["log_file"]
    batch_size = config["batch_size"]

    if task in cnf_tasks:
        data = CNFDataWrapper(root_dir, task, graph_type, log_file, batch_size)
    else:
        raise ValueError("Invalid task.")
    
    return data