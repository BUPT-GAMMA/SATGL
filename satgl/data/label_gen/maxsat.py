import os
import pandas
from tqdm import tqdm


who_path = os.path.join(os.path.dirname(__file__), "../../external/open-wbo/open-wbo")


def get_maxsat_from_file(cnf_path: str) -> list:
    r"""
    Get the max sat assignment from a file.

    Parameters
    ----------
    cnf_path : str
        The path to the cnf file.
    """
    solver_path = who_path
    maxsat_cmd = f"{solver_path} {cnf_path}"
    maxsat_info = os.popen(maxsat_cmd).readlines()

    is_find_optimum = False
    for line in maxsat_info:
        if "s OPTIMUM FOUND" in line:
            is_find_optimum = True
        if line.startswith("v"):
            max_sat_assignment = [int(int(x) > 0) for x in line.strip().strip("\n").split(" ")[1:]]
    
    assert(is_find_optimum == True)
    assert(len(max_sat_assignment) > 0)
    return max_sat_assignment

def gen_maxsat_label(cnf_dir: str, out_path: str) -> None:
    file_list = os.listdir(cnf_dir)
    name_list = []
    maxsat_list = []
    for file in tqdm(file_list):
        file_path = os.path.join(cnf_dir, file)
        maxsat = get_maxsat_from_file(file_path)
        name_list.append(file)
        maxsat_list.append(maxsat)

    label_df = pandas.DataFrame({'name': name_list, 'maxsat': maxsat_list})
    label_df.to_csv(out_path, index=False)