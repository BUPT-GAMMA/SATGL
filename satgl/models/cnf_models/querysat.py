import torch
import torch.nn as nn

from torch_geometric.nn.norm import PairNorm
from satgl.models.layers.mlp import MLP
from satgl.models.layers.cnf_conv import HeteroConv


class QuerySAT(nn.Module):
    """
        QuerySAT
    """
    def __init__(self, config):
        super(QuerySAT, self).__init__()
        self.config = config

        # check config
        if config["graph_type"] not in ["lcg"]:
            raise ValueError("QuerySAT only support lcg graph.")

        self.device = config.device
        self.emb_size = config.model_settings["emb_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.pad_size = config.model_settings["pad_size"]
        self.num_round = config.model_settings["num_round"]
        self.use_grad = config.model_settings["use_grad"]
        
        self.q_mlp = MLP(self.emb_size * 2 + self.pad_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        if self.use_grad:
            self.v_update = MLP(self.emb_size * 5, self.emb_size * 2, self.emb_size * 2, num_layer=self.num_fc)
        else:
            self.v_update = MLP(self.emb_size * 4, self.emb_size * 2, self.emb_size * 2, num_layer=self.num_fc)
        self.c_update = MLP(self.emb_size * 3, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.softplus = nn.Softplus()
        
        self.conv = HeteroConv()
        self.v_norm = PairNorm()
        self.c_norm = PairNorm()
    
    def forward(self, batch: dict) -> dict:
        g = batch["g"]
        l_pos_emb = g.nodes["pos_l"].data["emb"]
        l_neg_emb = g.nodes["neg_l"].data["emb"]
        l_emb = torch.cat([l_pos_emb, l_neg_emb], dim=0)
        c_emb = g.nodes["c"].data["emb"]

        num_variables = l_emb.shape[0] // 2

        for round_idx in enumerate(range(self.num_round)):
            # get query and clause loss
            v_emb = torch.cat(torch.chunk(l_emb, 2, dim=0), dim=1)
            noise_emb = torch.randn((num_variables, self.pad_size)).to(self.device)
            q_emb = self.q_mlp(torch.cat([v_emb, noise_emb], dim=1))
            pos_q_msg = self.softplus(q_emb)
            neg_q_msg = self.softplus(-q_emb)
            pos_q2c_msg = self.conv(g, "pos_l", "pos_l2c", "c", pos_q_msg)
            neg_q2c_msg = self.conv(g, "neg_l", "neg_l2c", "c", neg_q_msg)
            q2c_msg = pos_q2c_msg + neg_q2c_msg
            e_emb = torch.exp(-q2c_msg)

            # get variable grad
            if self.use_grad:
                q_emb.retain_grad()
                step_loss = torch.sum(e_emb)
                step_loss.backward(retain_graph=True)
                v_grad = q_emb.grad

            # literal message passing
            l_msg = l_emb
            pos_l_msg, neg_l_msg = torch.chunk(l_msg, 2, dim=0)
            pos_l2c_msg = self.conv(g, "pos_l", "pos_l2c", "c", pos_l_msg)
            neg_l2c_msg = self.conv(g, "neg_l", "neg_l2c", "c", neg_l_msg)
            l2c_msg = pos_l2c_msg + neg_l2c_msg

            # clause message passing
            c_msg = c_emb
            pos_c2l_msg = self.conv(g, "c", "pos_c2l", "pos_l", c_msg)
            neg_c2l_msg = self.conv(g, "c", "neg_c2l", "neg_l", c_msg)
            c2l_msg = torch.cat([pos_c2l_msg, neg_c2l_msg], dim=0)
            pos_l_emb, neg_l_emb = torch.chunk(l_emb, 2, dim=0)
            flip_l_emb = torch.cat([neg_l_emb, pos_l_emb], dim=0)

            # update
            if self.use_grad:
                v_emb = self.v_update(torch.cat([v_emb, pos_c2l_msg, neg_c2l_msg, v_grad], dim=1))
            else:
                v_emb = self.v_update(torch.cat([v_emb, pos_c2l_msg, neg_c2l_msg], dim=1))
            c_emb = self.c_update(torch.cat([l2c_msg, c_emb, e_emb], dim=1))
            
            # norm
            # v_emb = self.v_norm(v_emb)
            # c_emb = self.c_norm(c_emb)

            l_emb = torch.cat(torch.chunk(v_emb, 2, dim=1), dim=0)

        l_pos_emb, l_neg_emb = torch.chunk(l_emb, 2, dim=0)
        g.nodes["pos_l"].data["emb"] = l_pos_emb
        g.nodes["neg_l"].data["emb"] = l_neg_emb
        g.nodes["c"].data["emb"] = c_emb

        return batch
