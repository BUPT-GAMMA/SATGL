import os
import dgl
import torch


def parse_cnf_file(file_path):
    r"""
    Parse the cnf file, return the number of variables, number of clauses and the clause list.

    Parameters:
    -----------
        file_path (str): Path to the cnf file.

    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) < 1 or tokens[0] != 'p':
            i += 1
        else:
            break
    
    if i == len(lines):
        return 0, []
    
    header = lines[i].strip().split()
    num_variables = int(header[2])
    num_clauses = int(header[3])
    clause_list = []
    for line in lines[i+1:]:
        tokens = line.strip().split()
        if tokens[0] == 'c':
            continue
        
        clause = [int(s) for s in tokens[:-1]]
        clause_list.append(clause)
    return num_variables, num_clauses, clause_list


def build_hetero_lcg(num_variables, num_clauses, clause_list):
    r"""
    Build a heterogeneous graph for LCG.

    Parameters:
    -----------
        num_variables (int): Number of variables.
        num_clauses (int): Number of clauses.
        clause_list (list): List of clauses.

    """
    pos_src_list = []
    pos_dst_list = []
    neg_src_list = []
    neg_dst_list = []
    
    for c_id, c in enumerate(clause_list):
        for v in c:
            if v > 0:
                pos_src_list.append(v - 1)
                pos_dst_list.append(c_id)
            else:
                neg_src_list.append(-v - 1)
                neg_dst_list.append(c_id)
    
    edge_dict = {
        ("pos_l", "pos_l2c", "c"): (pos_src_list, pos_dst_list),
        ("neg_l", "neg_l2c", "c"): (neg_src_list, neg_dst_list),
        ("c", "pos_c2l", "pos_l"): (pos_dst_list, pos_src_list),
        ("c", "neg_c2l", "neg_l"): (neg_dst_list, neg_src_list),
    }
    
    num_node_dict = {
        "pos_l": num_variables,
        "neg_l": num_variables,
        "c": num_clauses,
    }
            
    g = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)
    return g

def build_hetero_lig(num_variables, num_clauses, clause_list):
    r"""
    Build a heterogeneous graph for LIG.

    Parameters:
    -----------
        num_variables (int): Number of variables.
        num_clauses (int): Number of clauses.
        clause_list (list): List of clauses.

    """
    pos_pos_src_list = []
    pos_pos_dst_list = []
    pos_neg_src_list = []
    pos_neg_dst_list = []
    neg_pos_src_list = []
    neg_pos_dst_list = []
    neg_neg_src_list = []
    neg_neg_dst_list = []


    for c_id, c in enumerate(clause_list):
        for i in range(len(c)):
            for j in range(i):
                if c[i] > 0 and c[j] > 0:
                    pos_pos_src_list.append(c[i] - 1)
                    pos_pos_dst_list.append(c[j] - 1)
                elif c[i] > 0 and c[j] < 0:
                    pos_neg_src_list.append(c[i] - 1)
                    pos_neg_dst_list.append(-c[j] - 1)
                elif c[i] < 0 and c[j] > 0:
                    neg_pos_src_list.append(-c[i] - 1)
                    neg_pos_dst_list.append(c[j] - 1)
                else:
                    neg_neg_src_list.append(-c[i] - 1)
                    neg_neg_dst_list.append(-c[j] - 1)
    
    edge_dict = {
        ("pos_l", "pos_pos_l2l", "pos_l"): (pos_pos_src_list, pos_pos_dst_list),
        ("pos_l", "pos_neg_l2l", "neg_l"): (pos_neg_src_list, pos_neg_dst_list),
        ("neg_l", "neg_pos_l2l", "pos_l"): (neg_pos_src_list, neg_pos_dst_list),
        ("neg_l", "neg_neg_l2l", "neg_l"): (neg_neg_src_list, neg_neg_dst_list),
    }

    num_node_dict = {
        "pos_l": num_variables,
        "neg_l": num_variables,
    }

    g = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)
    return g


def build_hetero_vcg(num_variables, num_clauses, clause_list):
    r"""
    Build a heterogeneous graph for VCG.

    Parameters:
    -----------
        num_variable (int): Number of variables.
        num_clause (int): Number of clauses.
        clause_list (list): List of clauses.

    """
    pos_src_list = []
    pos_dst_list = []
    neg_src_list = []
    neg_dst_list = []
    
    for c_id, c in enumerate(clause_list):
        for v in c:
            if v > 0:
                pos_src_list.append(v - 1)
                pos_dst_list.append(c_id)
            else:
                neg_src_list.append(-v - 1)
                neg_dst_list.append(c_id)
    
    edge_dict = {
        ("v", "v2c", "c"): (pos_src_list + neg_src_list, pos_dst_list + neg_dst_list),
        ("c", "c2v", "v"): (pos_dst_list + neg_dst_list, pos_src_list + neg_src_list),
    }
    
    num_node_dict = {
        "c": num_clauses,
        "v": num_variables
    }
            
    g = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)
    return g

def build_hetero_vig(num_variables, num_clauses, clause_list):
    r"""
    Build a heterogeneous graph for VIG.

    Parameters:
    -----------
        num_variable (int): Number of variables.
        num_clause (int): Number of clauses.
        clause_list (list): List of clauses.

    """
    src_list = []
    dst_list = []
    
    for c_id, c in enumerate(clause_list):
        for i in range(len(c)):
            for j in range(i):
                v1 = c[i] - 1 if c[i] > 0 else -c[i] - 1
                v2 = c[j] - 1 if c[j] > 0 else -c[j] - 1
                src_list.append(v1)
                dst_list.append(v2)
    
    edge_dict = {
        ("v", "v2v", "v"): (src_list, dst_list),
    }
    
    num_node_dict = {
        "v": num_variables
    }
            
    g = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)
    return g

# def build_homo_lcg(num_variable, num_clause, clause_list):
#     r"""
#     Build a homogeneous graph for LCG.

#     Parameters:
#     -----------
#         num_variable (int): Number of variables.
#         num_clause (int): Number of clauses.
#         clause_list (list): List of clauses.
#     """
#     num_nodes = num_variable * 2 + num_clause
#     src_list = []
#     dst_list = []

#     for c_id, c in enumerate(clause_list):
#         for v in c:
#             v_idx = v - 1 if v > 0 else -v - 1 + num_variable
#             c_idx = 2 * num_variable + c_id

#             src_list.append(v_idx)
#             dst_list.append(c_idx)
#             src_list.append(c_idx)
#             dst_list.append(v_idx)

#     g = dgl.graph((src_list, dst_list), num_nodes=num_nodes)
#     node_type = []
#     for node_id in range(2 * num_variable + num_clause):
#         if node_id < num_variable:
#             node_type.append(0)
#         elif node_id < 2 * num_variable:
#             node_type.append(1)
#         else:
#             node_type.append(2)
#     g.ndata["node_type"] = torch.tensor(node_type).float()
#     return g

# def build_homo_vcg(num_variable, num_clause, clause_list):
#     r"""
#     Build a homogeneous graph for VCG.

#     Parameters:
#     -----------
#         num_variable (int): Number of variables.
#         num_clause (int): Number of clauses.
#         clause_list (list): List of clauses.
#     """
#     src_list = []
#     dst_list = []

#     for c_id, c in enumerate(clause_list):
#         for v in c:
#             if v > 0:
#                 src_list.append(v - 1)
#             else:
#                 src_list.append(-v - 1)
#             dst_list.append(c_id + num_variable)

#     g = dgl.graph((src_list, dst_list))
#     node_type = []
#     for node_id in range(num_variable + num_clause):
#         if node_id < num_variable:
#             node_type.append(0)
#         else:
#             node_type.append(1)
#     g.ndata["node_type"] = torch.tensor(node_type).float()
#     return g

# def build_homo_vig(num_variable, num_clause, clause_list):
#     r"""
#     Build a homogeneous graph for VIG.

#     Parameters:
#     -----------
#         num_variable (int): Number of variables.
#         num_clause (int): Number of clauses.
#         clause_list (list): List of clauses.
#     """
#     src_list = []
#     dst_list = []

#     for c_id, c in enumerate(clause_list):
#         for i in range(len(c)):
#             for j in range(i):
#                 v1 = c[i] - 1 if c[i] > 0 else -c[i] - 1
#                 v2 = c[j] - 1 if c[j] > 0 else -c[j] - 1
#                 src_list.append(v1)
#                 dst_list.append(v2)

#     g = dgl.graph((src_list, dst_list))
#     return g

# def build_homo_lig(num_variable, num_clause, clause_list):
#     r"""
#     Build a homogeneous graph for LIG.

#     Parameters:
#     -----------
#         num_variable (int): Number of variables.
#         num_clause (int): Number of clauses.
#         clause_list (list): List of clauses.
#     """
#     src_list = []
#     dst_list = []

#     for c_id, c in enumerate(clause_list):
#         for i in range(len(c)):
#             for j in range(i):
#                 v1 = c[i] - 1 if c[i] > 0 else -c[i] - 1 + num_variable
#                 v2 = c[j] - 1 if c[j] > 0 else -c[j] - 1 + num_variable
#                 src_list.append(v1)
#                 dst_list.append(v2)

#     g = dgl.graph((src_list, dst_list))
#     node_type = []
#     for node_id in range(num_variable * 2):
#         if node_id < num_variable:
#             node_type.append(0)
#         else:
#             node_type.append(1)
#     g.ndata["node_type"] = torch.tensor(node_type).float()
#     return g