import os
import pandas
from tqdm import tqdm


picomus_path = os.path.join(os.path.dirname(__file__), "../../external/picosat-965/picomus")


def get_unsat_core_from_file(file_path):
    solver_path = picomus_path
    mus_cmd = solver_path + " " + file_path
    mus_info = os.popen(mus_cmd).readlines()

    # Parse MUS Info
    sat_flag = False
    for line in mus_info:
        if 's SATISFIABLE' in line:
            sat_flag = True
            break

    num_clause = 0
    with open(file_path, "r") as f:
        cnf = f.readlines()
        for line in cnf:
            if line.startswith("p cnf"):
                num_clause = int(line.split()[3])

    unsat_core = [0] * num_clause
    if not sat_flag:
        for line in mus_info:
            if 'v ' in line:
                ele = line.replace('v', '').replace(' ', '').replace('\n', '')
                ele = int(ele)
                if ele > 0:
                    unsat_core[ele - 1] = 1

    return unsat_core

def gen_unsat_core_label(cnf_dir, out_path):
    file_list = os.listdir(cnf_dir)
    name_list = []
    unsat_core_list = []
    for file in tqdm(file_list):
        file_path = os.path.join(cnf_dir, file)
        unsat_core = get_unsat_core_from_file(file_path)
        name_list.append(file)
        unsat_core_list.append(unsat_core)

    label_df = pandas.DataFrame({'name': name_list, 'unsat_core': unsat_core_list})
    label_df.to_csv(out_path, index=False)