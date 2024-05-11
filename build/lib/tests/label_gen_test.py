import os
import sys
import torch
import torch.nn as nn
import argparse

current_dir = os.path.dirname(__file__)
relative_path = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, relative_path)

from satgl.data.label_gen.maxsat import gen_maxsat_label
from satgl.data.label_gen.satisfiability import gen_satisfiability_label
from satgl.data.label_gen.unsat_core import gen_unsat_core_label


def maxsat_label_gen(cnf_dir):
    train_path = os.path.join(cnf_dir, "train")
    test_path = os.path.join(cnf_dir, "test")
    valid_path = os.path.join(cnf_dir, "valid")

    train_label_path = os.path.join(cnf_dir, "label", "train.csv")
    valid_label_path = os.path.join(cnf_dir, "label", "valid.csv")
    test_label_path = os.path.join(cnf_dir, "label", "test.csv")

    gen_maxsat_label(train_path, train_label_path)
    gen_maxsat_label(valid_path, valid_label_path)
    gen_maxsat_label(test_path, test_label_path)

def satisfiability_label_gen(cnf_dir):
    train_path = os.path.join(cnf_dir, "train")
    test_path = os.path.join(cnf_dir, "test")
    valid_path = os.path.join(cnf_dir, "valid")

    train_label_path = os.path.join(cnf_dir, "label", "train.csv")
    valid_label_path = os.path.join(cnf_dir, "label", "valid.csv")
    test_label_path = os.path.join(cnf_dir, "label", "test.csv")

    gen_satisfiability_label(train_path, train_label_path)
    gen_satisfiability_label(valid_path, valid_label_path)
    gen_satisfiability_label(test_path, test_label_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnf_dir", type=str, default="../benchmarks/sr")
    args = parser.parse_args()

    satisfiability_label_gen(args.cnf_dir)