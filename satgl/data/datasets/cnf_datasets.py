import dgl
import os
import torch
import random
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from dgl.data.dgl_dataset import DGLDataset

from typing import Optional
from satgl.config.config import Config
from satgl.data.cnf_utils import(
    parse_cnf_file,
    build_hetero_lcg,
    build_hetero_vcg,
    build_hetero_lig,
    build_hetero_vig
)


class CNFDataset(DGLDataset):
    r"""
    Abstract class for CNF dataset.
    """
    def __init__(self, cnf_dir: str, label_path:str , graph_type:str):
        self.cnf_dir = cnf_dir
        self.label_path = label_path
        self.graph_type = graph_type
        self.processed_data_dir = self.cnf_dir.strip("/") + "processed"
        super(CNFDataset, self).__init__(name="CNFDataset")

    
    def process(self):
        r"""
        Process the dataset.
        """
        raise NotImplementedError
    
    def load(self):
        r"""
        Load the dataset.
        """
        pass
        

    def save(self):
        r"""
        Save the dataset.
        """
        pass
    
    def build_graph(self, cnf_file_path: str) -> any:
        r"""
        Build various cnf graph.
        """
        num_variables, num_clauses, clause_list = parse_cnf_file(cnf_file_path)
        if self.graph_type == "lcg":
            return build_hetero_lcg(num_variables, num_clauses, clause_list)
        elif self.graph_type == "vcg":
            return build_hetero_vcg(num_variables, num_clauses, clause_list)
        elif self.graph_type == "lig":
            return build_hetero_lig(num_variables, num_clauses, clause_list)
        elif self.graph_type == "vig":
            return build_hetero_vig(num_variables, num_clauses, clause_list)
        else:
            raise ValueError("Invalid graph type.")
                
    def _get_info(self, num_variables, num_clauses, clause_list):
        return {
            "num_variables": num_variables,
            "num_clauses": num_clauses
        }

        
    def __getitem__(self, idx: int) -> any:
        r"""
        Get the item.
        """
        raise NotImplementedError

    def __len__(self):
        r"""
        Get the length.
        """
        raise NotImplementedError
    
    # def __getitem__(self, idx):
    #     if idx * 2 + 1 < len(self.data_list):
    #         item = [self.data_list[idx * 2], self.data_list[idx * 2 + 1]]
    #         random.shuffle(item)
    #         return item
    #     elif idx * 2 + 1 == len(self.data_list):
    #         return [self.data_list[idx * 2]]
    #     else:
    #         raise IndexError("Index out of range.")

    # def __len__(self):
    #     return (len(self.data_list) + 1) // 2
    
class SatistifiabilityDataset(CNFDataset):
    r"""
    Satisfiability task class for CNF dataset.
    """
    def __init__(self, cnf_dir:str, label_path:str, graph_type:str):
        super().__init__(cnf_dir, label_path, graph_type)
    
    def process(self):
        label_df = pd.read_csv(self.label_path, sep=',')

        # sort benchmark by label to balance the distribution of each batch.
        label_occur_times = {}
        self.data_list = []
        for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
            name = row['name']
            label = int(row['satisfiability'])
            cnf_path = os.path.join(self.cnf_dir, name)
            num_variable, num_clause, clause_list = parse_cnf_file(cnf_path)
            cnf_graph = self.build_graph(cnf_path)
            info = self._get_info(num_variable, num_clause, clause_list)
            
            if label not in label_occur_times:
                label_occur_times[label] = 0
            label_occur_times[label] += 1
            self.data_list.append({"g": cnf_graph, "label": label, "info": info, "sort_key":(label_occur_times[label], label)})
        self.data_list.sort(key=lambda x: x["sort_key"])

        # remove sort key
        self.data_list = [{"g": x["g"], "label": x["label"], "info": x["info"]} for x in self.data_list]
        
    
    def __getitem__(self, idx):
        if idx * 2 + 1 < len(self.data_list):
            item = [self.data_list[idx * 2], self.data_list[idx * 2 + 1]]
            random.shuffle(item)
            return item
        elif idx * 2 + 1 == len(self.data_list):
            return [self.data_list[idx * 2]]
        else:
            raise IndexError("Index out of range.")

    def __len__(self):
        return (len(self.data_list) + 1) // 2

class MaxSATDataset(CNFDataset):
    def __init__(self, cnf_dir:str, label_path:str, graph_type:str):
        super().__init__(cnf_dir, label_path, graph_type)

    def process(self):
        label_df = pd.read_csv(self.label_path, sep=',')
        self.data_list = []
        for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
            name = row['name']
            label = eval(row['maxsat'])
            cnf_path = os.path.join(self.cnf_dir, name)
            num_variable, num_clause, clause_list = parse_cnf_file(cnf_path)
            cnf_graph = self.build_graph(cnf_path)
            info = self._get_info(num_variable, num_clause, clause_list)
            self.data_list.append({"g": cnf_graph, "label": label, "info": info})

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

class UnSATCoreDataset(CNFDataset):
    def __init__(self, cnf_dir:str, label_path:str, graph_type:str):
        super().__init__(cnf_dir, label_path, graph_type)

    def process(self):
        label_df = pd.read_csv(self.label_path, sep=',')
        self.data_list = []
        for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
            name = row['name']
            label = eval(row['unsat_core'])
            cnf_path = os.path.join(self.cnf_dir, name)
            num_variable, num_clause, clause_list = parse_cnf_file(cnf_path)
            cnf_graph = self.build_graph(cnf_path)
            info = self._get_info(num_variable, num_clause, clause_list)
            self.data_list.append({"g": cnf_graph, "label": label, "info": info})

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)
