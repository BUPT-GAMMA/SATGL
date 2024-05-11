import dgl.function as fn
import numpy as np
import torch


from torch import nn
from torch.nn import functional as F, Parameter


class HeteroConv(nn.Module):
    r"""
        HeteroConv pass the src_node's feature to dst_node
    """
    def __init__(self):
        super(HeteroConv, self).__init__()

    def forward(self, g, src_type, e_type, dst_type, src_emb):
        r"""
        Aggregates the embeddings of the source nodes and updates the destination nodes.

        Parameters
        ----------
        g : DGLHeteroGraph
            The heterogeneous graph.
        src_type : str
            The source node type.
        e_type : str
            The edge type.
        dst_type : str
            The destination node type.
        src_emb : torch.Tensor
            The source node embeddings.

        """
        rel_g = g[src_type, e_type, dst_type]
        with rel_g.local_scope():
            rel_g.nodes[src_type].data["h"] = src_emb
            rel_g.apply_edges(fn.copy_u("h", "m"))
            rel_g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))
            dst_emb = rel_g.nodes[dst_type].data["h"]
        return dst_emb


class HeteroGCNConv(nn.Module):
    r"""
        HeteroGCNConv passes the src_node's feature to dst_node with normalization.
    """
    def __init__(self):
        super(HeteroGCNConv, self).__init__()

    def forward(self, g, src_type, e_type, dst_type, src_emb):
        r"""
        Use GCN to aggregate the embeddings of the source nodes and updates the destination nodes.(Normalization)

        Parameters
        ----------
        g : DGLHeteroGraph
            The heterogeneous graph.
        src_type : str
            The source node type.
        e_type : str
            The edge type.
        dst_type : str
            The destination node type.
        src_emb : torch.Tensor
            The source node embeddings.
        """
        rel_g = g[src_type, e_type, dst_type]
        with rel_g.local_scope():
            rel_g.nodes[src_type].data["h"] = src_emb

            # Compute degree matrix and its inverse square root
            src_deg_inv_sqrt = torch.pow(rel_g.out_degrees().float().clamp(min=1), -0.5)
            dst_deg_inv_sqrt = torch.pow(rel_g.in_degrees().float().clamp(min=1), -0.5)
            
            # message passing
            rel_g.nodes[src_type].data["h"] = src_emb * src_deg_inv_sqrt.unsqueeze(-1)
            rel_g.apply_edges(fn.copy_u("h", "m"))
            rel_g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))
            dst_emb = rel_g.nodes[dst_type].data["h"] * dst_deg_inv_sqrt.unsqueeze(-1)
        
        return dst_emb