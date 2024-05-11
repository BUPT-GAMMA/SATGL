import torch
import torch.nn as nn


from satgl.models.layers.mlp import MLP
from satgl.models.layers.cnf_conv import HeteroConv

class NeuroCore(nn.Module):
    def __init__(self, config):
        super(NeuroCore, self).__init__()
        self.config = config

         # check config
        if config["graph_type"] not in ["lcg"]:
            raise ValueError("NeuroCore only support lcg graph.")

        self.device = config.device
        self.emb_size = config.model_settings["emb_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.num_round = config.model_settings["num_round"]
        
        self.l_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.c_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.l_update = MLP(self.emb_size * 3, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.c_update = MLP(self.emb_size * 2, self.emb_size, self.emb_size, num_layer=self.num_fc)
        
        self.conv = HeteroConv()
    
    def forward(self, batch: dict) -> dict:
        g = batch["g"]
        l_pos_emb = g.nodes["pos_l"].data["emb"]
        l_neg_emb = g.nodes["neg_l"].data["emb"]
        l_emb = torch.cat([l_pos_emb, l_neg_emb], dim=0)
        c_emb = g.nodes["c"].data["emb"]

        for round_idx in enumerate(range(self.num_round)):
            # literal message passing
            l_msg = self.l_msg_mlp(l_emb)
            pos_l_msg, neg_l_msg = torch.chunk(l_msg, 2, dim=0)
            pos_l2c_msg = self.conv(g, "pos_l", "pos_l2c", "c", pos_l_msg)
            neg_l2c_msg = self.conv(g, "neg_l", "neg_l2c", "c", neg_l_msg)
            l2c_msg = pos_l2c_msg + neg_l2c_msg
            
            # clause message passing
            c_msg = self.c_msg_mlp(c_emb)
            pos_c2l_msg = self.conv(g, "c", "pos_c2l", "pos_l", c_msg)
            neg_c2l_msg = self.conv(g, "c", "neg_c2l", "neg_l", c_msg)
            c2l_msg = torch.cat([pos_c2l_msg, neg_c2l_msg], dim=0)
            pos_l_emb, neg_l_emb = torch.chunk(l_emb, 2, dim=0)
            flip_l_emb = torch.cat([neg_l_emb, pos_l_emb], dim=0)
            
            l_emb = self.l_update(torch.cat([c2l_msg, l_emb, flip_l_emb], dim=1))
            c_emb = self.c_update(torch.cat([l2c_msg, c_emb], dim=1))

        
        l_pos_emb, l_neg_emb = torch.chunk(l_emb, 2, dim=0)
        g.nodes["pos_l"].data["emb"] = l_pos_emb
        g.nodes["neg_l"].data["emb"] = l_neg_emb
        g.nodes["c"].data["emb"] = c_emb
        
        return batch