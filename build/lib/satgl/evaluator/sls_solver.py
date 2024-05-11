import os
import re
import subprocess
import tempfile
import time
import signal

sls_solver_root_path = os.path.join(os.path.dirname(__file__), "../external/sls_solvers")
solver_path = {
    "probsat": os.path.join(sls_solver_root_path, "probSAT", "probSAT"),
}

def parse_sls_solver_output(file_path):
    """
    Parse the output of the sls solver.

    Parameters
    ----------
    file_path : str
        The file path of the output file.

    Returns
    -------
    dict
        The parsed output.
    """

    # example of probSAT output
    # c numFlips                      : 0        
    # c avg. flips/variable           : 0.00    
    # c avg. flips/clause             : 0.00    
    # c flips/sec                     : -nan    
    # c CPU Time                      : 0.0000  
    num_flips_regex = re.compile(r"c numFlips\s+:\s+(\d+)")
    avg_flips_per_variable_regex = re.compile(r"c avg. flips/variable\s+:\s+([\d.]+)")
    avg_flips_per_clause_regex = re.compile(r"c avg. flips/clause\s+:\s+([\d.]+)")
    cpu_time_regex = re.compile(r"c CPU Time\s+:\s+([\d.]+)")

    with open(file_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            num_flip_match = num_flips_regex.match(line)
            if num_flip_match:
                num_flips = int(num_flip_match.group(1))
                continue

            avg_flips_per_variable_match = avg_flips_per_variable_regex.match(line)
            if avg_flips_per_variable_match:
                avg_flips_per_variable = float(avg_flips_per_variable_match.group(1))
                continue

            avg_flips_per_clause_match = avg_flips_per_clause_regex.match(line)
            if avg_flips_per_clause_match:
                avg_flips_per_clause = float(avg_flips_per_clause_match.group(1))
                continue

            cpu_time_match = cpu_time_regex.match(line)
            if cpu_time_match:
                cpu_time = float(cpu_time_match.group(1))
                continue
        
    return {
        "num_flips": num_flips,
        "avg_flips_per_variable": avg_flips_per_variable,
        "avg_flips_per_clause": avg_flips_per_clause,
        "cpu_time": cpu_time,
    }

def run_sls_solver(
        solver_name: str, 
        cnf_path: str, 
        init_vars_path: str=None,
        time_limit=60,
    ):
    """
    Run the sls solver.

    Parameters
    ----------
    solver_name : str
        The name of the solver.
    cnf_path : str
        The file path of the CNF file.
    init_vars_path : str
        The file path of the initial variables.
    time_limit : int
        The time limit in seconds.
    """

    if solver_name not in solver_path:
        raise ValueError(f"Solver {solver_name} is not supported.")

    solver = solver_path[solver_name]

    cmd = solver
    if init_vars_path is not None:
        cmd += f" -v {init_vars_path}"
    cmd += f" {cnf_path}"

    # run the solver and 1save the output to a temporary file
    output_file = tempfile.NamedTemporaryFile(delete=False)
    with open(output_file.name, "w") as f:
        is_timeout = False

        subp = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f, preexec_fn=os.setsid)
        try:
            subp.communicate(timeout=time_limit)
        except subprocess.TimeoutExpired:
            is_timeout = True
        finally:
            try:
                os.killpg(os.getpgid(subp.pid), signal.SIGKILL)
            except:
                pass  

    
    if is_timeout:
        result = {
            "num_flips": None,
            "avg_flips_per_variable": None,
            "avg_flips_per_clause": None,
            "flips_per_sec": None,
            "cpu_time": time_limit,
        }
    else:
        result = parse_sls_solver_output(output_file.name)
    
    # remove the output file
    os.remove(output_file.name)
    
    return result
        
            