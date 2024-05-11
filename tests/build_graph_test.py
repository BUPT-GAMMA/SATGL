import os
import sys

current_dir = os.path.dirname(__file__)
relative_path = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, relative_path)

from satgl.data.cnf_utils import(
    parse_cnf_file,
    build_hetero_lcg,
    build_hetero_vcg,
    build_homo_lcg,
    build_homo_lig,
    build_homo_vcg,
    build_homo_vig
)

if __name__ == "__main__":
    # parse cnf file
    num_variables, num_clauses, clause_list = parse_cnf_file(
        "./test_benchmarks/test_cnf/test.cnf"
    )

    # build hetero lcg
    build_hetero_lcg(num_variables, num_clauses, clause_list)
    build_hetero_vcg(num_variables, num_clauses, clause_list)
    build_homo_lcg(num_variables, num_clauses, clause_list)
    build_homo_vcg(num_variables, num_clauses, clause_list)
    build_homo_lig(num_variables, num_clauses, clause_list)
    build_homo_vig(num_variables, num_clauses, clause_list)




    