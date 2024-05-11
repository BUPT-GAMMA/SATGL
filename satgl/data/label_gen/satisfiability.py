import os
import pandas
import pysat

from tqdm import tqdm
from pysat.formula import CNF
from pysat.solvers import Solver

def get_satisfiability_from_file(file_path) -> int:
    """
    Get the satisfiability of a CNF file

    Parameters
    ----------
    file_path : str
        Path to the CNF file
    
    Returns
    -------
    int
        1 if the CNF is satisfiable, 0 otherwise
    """
    formula = CNF(from_file=file_path)
    solver = Solver()
    solver.append_formula(formula)
    return int(solver.solve())

def gen_satisfiability_label(cnf_dir: str, out_path: str) -> None:
    """
    Generate a CSV file with the satisfiability of each CNF file in a directory

    Parameters
    ----------
    cnf_dir : str
        Path to the directory containing the CNF files
    out_path : str
        Path to the output CSV file
    """
    file_list = os.listdir(cnf_dir)
    name_list = []
    satisfiability_list = []
    for file in tqdm(file_list):
        file_path = os.path.join(cnf_dir, file)
        satisfiability = get_satisfiability_from_file(file_path)
        name_list.append(file)
        satisfiability_list.append(satisfiability)

    label_df = pandas.DataFrame({'name': name_list, 'satisfiability': satisfiability_list})
    label_df.sort_values(by='name', inplace=True)
    label_df.to_csv(out_path, index=False)