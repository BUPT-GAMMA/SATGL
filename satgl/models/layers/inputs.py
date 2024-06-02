import torch
import dgl
import torch.nn as nn

from satgl.wrappers.model_wrappers.base_model_wrapper import ModelWrapper
from satgl.models.layers.mlp import MLP

class CNFPreStage(nn.Module):
    def __init__(
        self, 
        emb_size:int, 
        graph_type:str,
        num_fc:int = 3
    ) -> None:
        super(CNFPreStage, self).__init__()
        self.emb_size = emb_size
        self.graph_type = graph_type
        self.num_fc = num_fc

        assert(graph_type in ["lcg", "lig", "vig", "vcg"])
        if graph_type == "lcg" or graph_type == "lig":
            self.l_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)
            self.l_init_emb = nn.Parameter(torch.randn(emb_size))
        if graph_type == "vcg" or graph_type == "vig":
            self.v_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)
            self.v_init_emb = nn.Parameter(torch.randn(emb_size))
        if graph_type == "lcg" or graph_type == "lig":
            self.c_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)
            self.c_init_emb = nn.Parameter(torch.randn(emb_size))

    def forward(self, batch: dict) -> dict:
        g = batch["g"]
        if self.graph_type == "lcg" or self.graph_type == "lig":
            l_emb = self.l_init_mlp(self.l_init_emb)
            g.nodes["pos_l"].data["emb"] = l_emb.repeat(g.num_nodes("pos_l"), 1)
            g.nodes["neg_l"].data["emb"] = l_emb.repeat(g.num_nodes("neg_l"), 1)
        if self.graph_type == "vig" or self.graph_type == "vcg":
            v_emb = self.v_init_mlp(self.v_init_emb)
            g.nodes["v"].data["emb"] = v_emb.repeat(g.num_nodes("v"), 1)
        if self.graph_type == "lcg" or self.grapy_type == "vcg":
            c_emb = self.c_init_mlp(self.c_init_emb)
            g.nodes["c"].data["emb"] = c_emb.repeat(g.num_nodes("c"), 1)

        return batch
    
def get_pre_stage(
        emb_size:int, 
        graph_type:str,
        num_fc:int = 3,
        **kwargs
    ) -> nn.Module:
    return CNFPreStage(
        emb_size,
        graph_type,
        num_fc
    )