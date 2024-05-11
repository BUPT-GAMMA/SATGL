import os
import sys
import torch
import torch.nn as nn
import tempfile
import argparse
import random
import multiprocessing

from tqdm import tqdm
from multiprocessing import Process, Manager, Pool

current_dir = os.path.dirname(__file__)
relative_path = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, relative_path)

from satgl.evaluator.sls_solver import run_sls_solver
from satgl.data.cnf_utils import parse_cnf_file, build_hetero_lcg
from satgl.wrappers.get_model import get_model
from satgl.config.config import Config


convince = 0.7

def run_tests():
    cnf_path = os.path.join(os.path.dirname(__file__), "sls_test_cnf", "test.cnf")
    vars_path = os.path.join(os.path.dirname(__file__), "sls_test_cnf", "vars.txt")

    # random initial assignment
    results = run_sls_solver("probsat", cnf_path)
    print("random initial assignment results:")
    print(results)

    # with initial assignment
    results = run_sls_solver("probsat", cnf_path, init_vars_path=vars_path)
    print("with initial assignment results:")
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnf_dir", type=str, default="../benchmarks/SAT2022_mini_only_sat")
    parser.add_argument("--num_process", type=int, default=10)
    parser.add_argument("--time_limit", type=int, default=60)
    args = parser.parse_args()

    # load model
    neurosat_model_path = os.path.join(os.path.dirname(__file__), "save_model", "neurosat", "best_model.pth")

    config_files = ["./test_yaml/neurosat.yaml"]
    parameter_dict = {
        "task": "maxsat",
        "model": "neurosat",
        "dataset_path": "../benchmarks/sr",
        "device": "cuda:7"
    }
    config = Config(config_files, parameter_dict)
    model = get_model(config)
    model.load_state_dict(torch.load(neurosat_model_path))
    model = model.to(config["device"])

    time_limit = args.time_limit

    # defiine the file list
    file_list = os.listdir(args.cnf_dir)

    # remove unsat file cause probSAT can only solve sat problem
    file_list = [file for file in file_list if "unsat" not in file]

    # get model predict
    print("preprocessing model predict...")
    file_to_vars = {}
    for file in tqdm(file_list):
        cnf_path = os.path.join(args.cnf_dir, file)

        # model predict
        num_variables, num_clauses, clause_list = parse_cnf_file(cnf_path)
        hetero_graph = build_hetero_lcg(num_variables, num_clauses, clause_list)
        hetero_graph = hetero_graph.to(config["device"])

        input = {"g": hetero_graph}
        output = model(input)["output"].cpu().detach().numpy()

        vars_path = tempfile.NamedTemporaryFile(delete=False)
        file_to_vars[file] = vars_path.name
        with open(vars_path.name, "w") as f:
            for i, val in enumerate(output):
                if val > convince:
                    f.write(f"{i+1} ")
                elif val < 1 - convince:
                    f.write(f"-{i+1} ")
                else:
                    # rand a value 0 or 1
                    random_val = random.randint(0, 1)
                    if random_val == 0:
                        f.write(f"-{i+1} ")
                    else:
                        f.write(f"{i+1} ")

    # divide the file list into num_process parts
    num_process = args.num_process
    process_file_list = [[] for _ in range(num_process)]
    for i, file in enumerate(file_list):
        process_file_list[i % num_process].append(file)
    
    # shared variables
    shared_counter = multiprocessing.Value("i", 0)
    
    # define process worker
    def worker(worker_args):
        process_file_list, file_to_vars, time_limit = worker_args

        result_list = []
        
        for file in process_file_list:
            file_path = os.path.join(args.cnf_dir, file)
            if file_to_vars is not None:
                vars_path = file_to_vars[file]
                results = run_sls_solver("probsat", file_path, init_vars_path=vars_path, time_limit=time_limit)
                result_list.append([file, 1, results])
            else:
                results = run_sls_solver("probsat", file_path, time_limit=time_limit)
                result_list.append([file, 0, results])

            with shared_counter.get_lock():
                shared_counter.value += 1
                print(f"[{shared_counter.value}/{2 * len(file_list)}] : {file}")

        return result_list
    
    # final results
    final_result_list = []

    # start with model predict process
    with Pool(processes=num_process) as pool:  
        data = [(process_file_list[i], file_to_vars, time_limit) for i in range(num_process)] 
        results = pool.map(worker, data)
        final_result_list = []
        for result in results:
            final_result_list.extend(result)

    
    # start with random initial assignment process
    with Pool(processes=num_process) as pool:  
        data = [(process_file_list[i], None, time_limit) for i in range(num_process)] 
        results = pool.map(worker, data)
        for result in results:
            final_result_list.extend(result)


    # parse result
    random_flips = 0
    random_cpu_time = 0
    random_solved = 0
    model_flips = 0
    model_cpu_time = 0
    model_solved = 0
    for item in final_result_list:
        type = item[1]
        if type == 0:
            if item[2]["num_flips"] is not None:
                random_flips += item[2]["num_flips"]
                random_cpu_time += item[2]["cpu_time"]
                random_solved += 1
        else:
            if item[2]["num_flips"] is not None:
                model_flips += item[2]["num_flips"]
                model_cpu_time += item[2]["cpu_time"]
                model_solved += 1
    
    # delete predict file
    for file, vars_path in file_to_vars.items():
        os.remove(vars_path)

    # output results with table format
    random_avg_flips = random_flips / random_solved if random_solved != 0 else 0
    random_avg_cpu_time = random_cpu_time / random_solved if random_solved != 0 else 0
    model_avg_flips = model_flips / model_solved if model_solved != 0 else 0
    model_avg_cpu_time = model_cpu_time / model_solved if model_solved != 0 else 0

    all_results = {
        "flips" : f"{random_flips} - {model_flips}",
        "avg_flips" : f"{random_avg_flips} - {model_avg_flips}",
        "cpu_time" : f"{random_cpu_time} - {model_cpu_time}",
        "avg_cpu_time" : f"{random_avg_cpu_time} - {model_avg_cpu_time}",
        "solved" : f"  {random_solved} - {model_solved}"
    }
    
    max_key_len = max(len(str(key)) for key in all_results.keys())
    max_val_len = max(len(str(val)) for val in all_results.values())
    max_key_len = max(max_key_len, len('metric')) + 4
    max_val_len = max(max_val_len, len('result')) + 4

    header = '|' + 'key'.ljust(max_key_len) + '|' + 'value'.ljust(max_val_len) + '|'
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    
    for key, val in all_results.items():
        row = '|' + str(key).ljust(max_key_len) + '|' + str(val).ljust(max_val_len) + '|'
        print(row)

    print('-' * len(header))
