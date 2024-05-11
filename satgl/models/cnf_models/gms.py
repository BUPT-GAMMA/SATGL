import torch
import torch.nn as nn


from satgl.models.layers.mlp import MLP
from satgl.models.layers.cnf_conv import HeteroConv

class GMS(nn.Module):
    """
        GMS from Can Graph Neural Networks Learn to Solve MaxSAT Problem?

    """
    def __init__(self, config):
        super(GMS, self).__init__()
        self.config = config
        
        if config["graph_type"] not in ["lcg"]:
            raise ValueError("GMS only support lcg graph.")

        self.device = config.device
        self.emb_size = config.model_settings["emb_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.num_round = config.model_settings["num_round"]
        
        self.pos_l_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.neg_l_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.pos_c_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.neg_c_msg_mlp = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.l_update = nn.LSTM(self.emb_size, self.emb_size)
        self.c_update = nn.LSTM(self.emb_size, self.emb_size)
        
        self.conv = HeteroConv()
    
    def forward(self, batch: dict) -> dict:
        g = batch["g"]
        l_pos_emb = g.nodes["pos_l"].data["emb"]
        l_neg_emb = g.nodes["neg_l"].data["emb"]
        l_emb = torch.cat([l_pos_emb, l_neg_emb], dim=0)
        c_emb = g.nodes["c"].data["emb"]

        num_literals = l_emb.shape[0]
        num_clauses = c_emb.shape[0]
            
        l_state = (l_emb.reshape(1, num_literals, -1), torch.zeros(1, num_literals, self.emb_size).to(self.device))
        c_state = (c_emb.reshape(1, num_clauses, -1), torch.zeros(1, num_clauses, self.emb_size).to(self.device))
        
        for round_idx in enumerate(range(self.num_round)):
            # literal message passing
            l_hidden = l_state[0].squeeze(0)
            pos_l_hidden, neg_l_hidden = torch.chunk(l_hidden, 2, dim=0)
            pos_l_msg = self.pos_l_msg_mlp(pos_l_hidden)
            neg_l_msg = self.neg_l_msg_mlp(neg_l_hidden)
            pos_l2c_msg = self.conv(g, "pos_l", "pos_l2c", "c", pos_l_msg)
            neg_l2c_msg = self.conv(g, "neg_l", "neg_l2c", "c", neg_l_msg)
            l2c_msg = pos_l2c_msg + neg_l2c_msg
            
            # clause message passing
            c_hidden = c_state[0].squeeze(0)
            pos_c_msg = self.pos_c_msg_mlp(c_hidden)
            neg_c_msg = self.neg_c_msg_mlp(c_hidden)
            pos_c2l_msg = self.conv(g, "c", "pos_c2l", "pos_l", pos_c_msg)
            neg_c2l_msg = self.conv(g, "c", "neg_c2l", "neg_l", neg_c_msg)
            c2l_msg = torch.cat([pos_c2l_msg, neg_c2l_msg], dim=0)
            flip_c2l_msg = torch.cat([neg_c2l_msg, pos_c2l_msg], dim=0)
            
            _, l_state = self.l_update((c2l_msg + flip_c2l_msg).unsqueeze(0), l_state)
            _, c_state = self.c_update(l2c_msg.unsqueeze(0), c_state)

        
        l_emb = l_state[0].squeeze(0)
        l_pos_emb, l_neg_emb = torch.chunk(l_emb, 2, dim=0)
        g.nodes["pos_l"].data["emb"] = l_pos_emb
        g.nodes["neg_l"].data["emb"] = l_neg_emb
        g.nodes["c"].data["emb"] = c_state[0].squeeze(0)

        return batch
