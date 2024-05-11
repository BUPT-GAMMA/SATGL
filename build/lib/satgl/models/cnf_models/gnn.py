import torch
import dgl
import torch.nn as nn


from satgl.models.layers.mlp import MLP

from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch import HeteroGraphConv


class GeneralGNN(nn.Module):
    def __init__(self, config):
        r"""
        General GNN model for CNF problems.

        Parameters
        ----------
        config : Config
            Configuration object.
        
        The subclass must implement the set_conv method to set the convolution function

        """
        super(GeneralGNN, self).__init__()
        self.config = config

        # check config
        if config["graph_type"] not in ["lcg", "vcg", "lig", "vig"]:
            raise ValueError("General GNN only support lcg, lig, vcg, vig.")

        self.device = config.device
        self.emb_size = config.model_settings["emb_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.num_round = config.model_settings["num_round"]
        self.graph_type = config["graph_type"]

        # build network
        self.build_update_fn()
        self.set_forward_fn()
        self.set_conv()
        self.build_conv_layer()

    def set_conv(self):
        raise NotImplementedError
    
    def build_conv_layer(self):
        if self.graph_type == "lcg":
            self.conv_layer = HeteroGraphConv(
                {
                    ("pos_l", "pos_l2c", "c"): self.conv,
                    ("neg_l", "neg_l2c", "c"): self.conv,
                    ("c", "pos_c2l", "pos_l"): self.conv,
                    ("c", "neg_c2l", "neg_l"): self.conv,
                }, 
                aggregate="sum"
            )
        elif self.graph_type == "vcg":
            self.conv_layer = HeteroGraphConv(
                {
                    ("v", "v2c", "c"): self.conv,
                    ("c", "c2v", "v"): self.conv,
                }, 
                aggregate="sum"
            )
        elif self.graph_type == "lig":
            self.conv_layer = HeteroGraphConv(
                {
                    ("pos_l", "pos_pos_l2l", "pos_l"): self.conv,
                    ("pos_l", "pos_neg_l2l", "neg_l"): self.conv,
                    ("neg_l", "neg_pos_l2l", "pos_l"): self.conv,
                    ("neg_l", "neg_neg_l2l", "neg_l"): self.conv,
                }, 
                aggregate="sum"
            )
        elif self.graph_type == "vig":
            self.conv_layer = HeteroGraphConv(
                {
                    ("v", "v2v", "v"): self.conv,
                }, 
                aggregate="sum"
            )

    def build_update_fn(self):
        r"""
        Build update functions for the model.

        lcg uses l_msg_mlp, l_update, c_update
        vcg uses v_msg_mlp, c_msg_mlp, v_update, c_update
        lig uses l_msg_mlp, l_update
        vig uses v_msg_mlp, v_update
        """
        if self.graph_type == "lcg":
            self.l_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
            self.c_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
            self.l_update = MLP(self.emb_size * 3, self.emb_size, self.emb_size, num_layer=self.num_fc)
            self.c_update = MLP(self.emb_size * 2, self.emb_size, self.emb_size, num_layer=self.num_fc)
        elif self.graph_type == "vcg":
            self.v_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
            self.c_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
            self.v_update = MLP(self.emb_size * 2, self.emb_size, self.emb_size, num_layer=self.num_fc)
            self.c_update = MLP(self.emb_size * 2, self.emb_size, self.emb_size, num_layer=self.num_fc)
        elif self.graph_type == "vig":
            self.v_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
            self.v_update = MLP(self.emb_size * 2, self.emb_size, self.emb_size, num_layer=self.num_fc)
        elif self.graph_type == "lig":
            self.l_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
            self.l_update = MLP(self.emb_size * 3, self.emb_size, self.emb_size, num_layer=self.num_fc)
        else:
            raise ValueError("Invalid graph type")

    def set_forward_fn(self):
        if self.graph_type == "lcg":
            self.forward = self.lcg_forward
        elif self.graph_type == "vcg":
            self.forward = self.vcg_forward
        elif self.graph_type == "vig":
            self.forward = self.vig_forward
        elif self.graph_type == "lig":
            self.forward = self.lig_forward
        else:
            raise ValueError("Invalid graph type")
        
    def lcg_forward(self, batch: dict) -> dict:
        g = batch["g"]
        l_pos_emb = g.nodes["pos_l"].data["emb"]
        l_neg_emb = g.nodes["neg_l"].data["emb"]
        c_emb = g.nodes["c"].data["emb"]

        for round_idx in enumerate(range(self.num_round)):
            l_pos_msg = self.l_msg_mlp(l_pos_emb)
            l_neg_msg = self.l_msg_mlp(l_neg_emb)
            c_msg = self.c_msg_mlp(c_emb)

            msg = {
                "pos_l": l_pos_msg,
                "neg_l": l_neg_msg,
                "c": c_msg
            }
            conv_emb = self.conv_layer(g, msg)
            
            pos_cat_emb = torch.cat([conv_emb["pos_l"], l_pos_emb, l_neg_emb], dim=1)
            neg_cat_emb = torch.cat([conv_emb["neg_l"], l_neg_emb, l_pos_emb], dim=1)
            c_cat_emb = torch.cat([conv_emb["c"], c_emb], dim=1)

            l_pos_emb = self.l_update(pos_cat_emb)
            l_neg_emb = self.l_update(neg_cat_emb)
            c_emb = self.c_update(c_cat_emb)

        g.nodes["pos_l"].data["emb"] = l_pos_emb
        g.nodes["neg_l"].data["emb"] = l_neg_emb
        g.nodes["c"].data["emb"] = c_emb
        
        return batch
    
    def vcg_forward(self, batch: dict) -> dict:
        g = batch["g"]
        v_emb = g.nodes["v"].data["emb"]
        c_emb = g.nodes["c"].data["emb"]

        for round_idx in enumerate(range(self.num_round)):
            v_msg = self.v_msg_mlp(v_emb)
            c_msg = self.c_msg_mlp(c_emb)

            msg = {
                "v": v_msg,
                "c": c_msg
            }
            conv_emb = self.conv_layer(g, msg)
            
            v_cat_emb = torch.cat([conv_emb["v"], v_emb], dim=1)
            c_cat_emb = torch.cat([conv_emb["c"], c_emb], dim=1)

            v_emb = self.v_update(v_cat_emb)
            c_emb = self.c_update(c_cat_emb)

        g.nodes["v"].data["emb"] = v_emb
        g.nodes["c"].data["emb"] = c_emb
        
        return batch
    
    def lig_forward(self, batch: dict) -> dict:
        g = batch["g"]
        l_pos_emb = g.nodes["pos_l"].data["emb"]
        l_neg_emb = g.nodes["neg_l"].data["emb"]

        for round_idx in enumerate(range(self.num_round)):
            l_pos_msg = self.l_msg_mlp(l_pos_emb)
            l_neg_msg = self.l_msg_mlp(l_neg_emb)

            msg = {
                "pos_l": l_pos_msg,
                "neg_l": l_neg_msg,
            }
            conv_emb = self.conv_layer(g, msg)
            
            pos_cat_emb = torch.cat([conv_emb["pos_l"], l_pos_emb, l_neg_emb], dim=1)
            neg_cat_emb = torch.cat([conv_emb["neg_l"], l_neg_emb, l_pos_emb], dim=1)

            l_pos_emb = self.l_update(pos_cat_emb)
            l_neg_emb = self.l_update(neg_cat_emb)

        g.nodes["pos_l"].data["emb"] = l_pos_emb
        g.nodes["neg_l"].data["emb"] = l_neg_emb
        
        return batch

    def vig_forward(self, batch: dict) -> dict:
        g = batch["g"]
        v_emb = g.nodes["v"].data["emb"]

        for round_idx in enumerate(range(self.num_round)):
            v_msg = self.v_msg_mlp(v_emb)

            msg = {
                "v": v_msg,
            }
            conv_emb = self.conv_layer(g, msg)
            
            v_cat_emb = torch.cat([conv_emb["v"], v_emb], dim=1)

            v_emb = self.v_update(v_cat_emb)

        g.nodes["v"].data["emb"] = v_emb
        
        return batch
    
    def forward(self, batch: dict) -> dict:
        return self.forward(batch)

class GCN(GeneralGNN):
    def __init__(self, config):
        super(GCN, self).__init__(config)

    def set_conv(self):
        self.conv = GraphConv(self.emb_size, self.emb_size, allow_zero_in_degree=True)

class GIN(GeneralGNN):
    def __init__(self, config):
        super(GIN, self).__init__(config)

    def set_conv(self):
        self.conv = GINConv()
