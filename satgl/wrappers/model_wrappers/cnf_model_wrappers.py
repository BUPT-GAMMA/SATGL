import torch
import dgl
import torch.nn as nn

from satgl.wrappers.model_wrappers.base_model_wrapper import ModelWrapper
from satgl.models.layers.mlp import MLP


class CNFModelWrapper(ModelWrapper):
    def __init__(self, config) -> None:
        super(ModelWrapper, self).__init__()
        self.config = config
        self.init_pre_stage_networks()

    def init_pre_stage_networks(self):
        emb_size = self.config.model_settings["emb_size"]
        num_fc = self.config.model_settings["num_fc"]
        if self.config["graph_type"] == "lcg":
            self.l_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)
            self.c_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)

            self.l_init_emb = nn.Parameter(torch.randn(emb_size))
            self.c_init_emb = nn.Parameter(torch.randn(emb_size))
        elif self.config["graph_type"] == "vcg":
            self.v_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)
            self.c_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)
            
            self.v_init_emb = nn.Parameter(torch.randn(emb_size))
            self.c_init_emb = nn.Parameter(torch.randn(emb_size))
        elif self.config["graph_type"] == "lig":
            self.l_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)

            self.l_init_emb = nn.Parameter(torch.randn(emb_size))
        elif self.config["graph_type"] == "vig":
            self.v_init_mlp = MLP(emb_size, emb_size, emb_size, num_layer=num_fc)
            
            self.v_init_emb = nn.Parameter(torch.randn(emb_size))
        else:
            raise ValueError("Graph type not supported.")
    
    def init_post_stage_network(self):
        pass

    def pre_stage(self, batch: dict):
        g = batch["g"]
        if self.config["graph_type"] == "lcg":
            l_emb = self.l_init_mlp(self.l_init_emb)
            c_emb = self.c_init_mlp(self.c_init_emb)
            g.nodes["pos_l"].data["emb"] = l_emb.repeat(g.num_nodes("pos_l"), 1)
            g.nodes["neg_l"].data["emb"] = l_emb.repeat(g.num_nodes("neg_l"), 1)
            g.nodes["c"].data["emb"] = c_emb.repeat(g.num_nodes("c"), 1)
        elif self.config["graph_type"] == "vcg":
            v_emb = self.v_init_mlp(self.v_init_emb)
            c_emb = self.c_init_mlp(self.c_init_emb)
            g.nodes["v"].data["emb"] = v_emb.repeat(g.num_nodes("v"), 1)
            g.nodes["c"].data["emb"] = c_emb.repeat(g.num_nodes("c"), 1)
        elif self.config["graph_type"] == "lig":
            l_emb = self.l_init_mlp(self.l_init_emb)
            g.nodes["pos_l"].data["emb"] = l_emb.repeat(g.num_nodes("pos_l"), 1)
            g.nodes["neg_l"].data["emb"] = l_emb.repeat(g.num_nodes("neg_l"), 1)
        elif self.config["graph_type"] == "vig":
            v_emb = self.v_init_mlp(self.v_init_emb)
            g.nodes["v"].data["emb"] = v_emb.repeat(g.num_nodes("v"), 1)
        else:
            raise ValueError("Graph type not supported.")

        return batch
    
    def lcg_post_stage(self, batch: dict):
        pass

    def vcg_post_stage(self, batch: dict):
        pass

    def lig_post_stage(self, batch: dict):
        pass

    def vig_post_stage(self, batch: dict):
        pass

    def post_stage(self, batch: dict):
        pass

    def forward(self, batch: dict):
        pre_stage_out = self.pre_stage(batch)
        model_out = self.model(pre_stage_out)
        post_stage_out = self.post_stage(model_out)
        return post_stage_out
    

class SatisfiabilityModelWrapper(CNFModelWrapper):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.init_post_stage_network()

    def init_post_stage_network(self):
        emb_size = self.config.model_settings["emb_size"]
        num_fc = self.config.model_settings["num_fc"]
        graph_type = self.config["graph_type"]
        if graph_type == "lcg":
            self.output_mlp = MLP(emb_size * 2, emb_size, 1, num_layer=num_fc)
        else:
            raise ValueError("Graph type not supported.")

    def lcg_post_stage(self, batch: dict):
        g = batch["g"]
        num_variables = batch["info"]["num_variables"]

        l_pos_emb = g.nodes["pos_l"].data["emb"]
        l_neg_emb = g.nodes["neg_l"].data["emb"]
        c_emb = g.nodes["c"].data["emb"]
        g.ndata.pop("emb")

        v_emb = torch.cat([l_pos_emb, l_neg_emb], dim=1)
        pool_emb = dgl.ops.segment_reduce(num_variables, v_emb, reducer="mean")
        batch["output"] = torch.sigmoid(self.output_mlp(pool_emb)).squeeze(-1)
        return batch
    
    def lig_post_stage(self, batch: dict):
        g = batch["g"]
        num_variables = batch["info"]["num_variables"]

        l_emb = g.nodes["l"].data["emb"]
        c_emb = g.nodes["c"].data["emb"]

        pos_l_emb, neg_l_emb = torch.chunk(l_emb, 2, dim=0)
        v_emb = torch.cat([pos_l_emb, neg_l_emb], dim=0)
        pool_emb = dgl.ops.segment_reduce(num_variables, v_emb, reducer="mean")
        batch["output"] = torch.sigmoid(self.output_mlp(pool_emb)).squeeze(-1)
        return batch
    
    def vcg_post_stage(self, batch: dict):
        g = batch["g"]

        v_emb = g.nodes["v"].data["emb"]

        batch["output"] = torch.sigmoid(self.output_mlp(v_emb)).squeeze(-1)
        return batch

    def vig_post_stage(self, batch: dict):
        g = batch["g"]

        v_emb = g.nodes["v"].data["emb"]

        batch["output"] = torch.sigmoid(self.output_mlp(v_emb)).squeeze(-1)
        return batch


    def post_stage(self, batch: dict):
        task = self.config["task"]
        graph_type = self.config["graph_type"]

        if graph_type == "lcg":
            return self.lcg_post_stage(batch)
        else:
            raise ValueError("Graph type not supported.")
        
            
class MaxSATModelWrapper(CNFModelWrapper):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.init_post_stage_network()

    def init_post_stage_network(self):
        emb_size = self.config.model_settings["emb_size"]
        num_fc = self.config.model_settings["num_fc"]
        graph_type = self.config["graph_type"]

        if graph_type == "lcg":
            self.output_mlp = MLP(emb_size * 2, emb_size, 1, num_layer=num_fc)
        elif graph_type == "lig":
            self.output_mlp = MLP(emb_size * 2, emb_size, 1, num_layer=num_fc)
        elif graph_type == "vcg":
            self.output_mlp = MLP(emb_size, emb_size, 1, num_layer=num_fc)
        elif graph_type == "vig":
            self.output_mlp = MLP(emb_size, emb_size, 1, num_layer=num_fc)
        else:
            raise ValueError("Graph type not supported.")
    
    def lcg_post_stage(self, batch: dict):
        g = batch["g"]

        l_pos_emb = g.nodes["pos_l"].data["emb"]
        l_neg_emb = g.nodes["neg_l"].data["emb"]

        v_emb = torch.cat([l_pos_emb, l_neg_emb], dim=1)
        batch["output"] = torch.sigmoid(self.output_mlp(v_emb)).squeeze(-1)
        return batch
    
    def lig_post_stage(self, batch: dict):
        g = batch["g"]

        l_pos_emb = g.nodes["pos_l"].data["emb"]
        l_neg_emb = g.nodes["neg_l"].data["emb"]

        v_emb = torch.cat([l_pos_emb, l_neg_emb], dim=1)
        batch["output"] = torch.sigmoid(self.output_mlp(v_emb)).squeeze(-1)
        return batch

    def post_stage(self, batch: dict):
        task = self.config["task"]
        graph_type = self.config["graph_type"]

        if graph_type == "lcg":
            return self.lcg_post_stage(batch)
        elif graph_type == "lig":
            return self.lig_post_stage(batch)
        else:
            raise ValueError("Graph type not supported.")
        

class UnSATCoreModelWrapper(CNFModelWrapper):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.init_post_stage_network()

    def init_post_stage_network(self):
        emb_size = self.config.model_settings["emb_size"]
        num_fc = self.config.model_settings["num_fc"]
        graph_type = self.config["graph_type"]

        if graph_type == "lcg":
            self.output_mlp = MLP(emb_size, emb_size, 1, num_layer=num_fc)
        else:
            raise ValueError("Graph type not supported.")
    
    def lcg_post_stage(self, batch: dict):
        g = batch["g"]

        c_emb = g.nodes["c"].data["emb"]

        batch["output"] = torch.sigmoid(self.output_mlp(c_emb)).squeeze(-1)
        return batch
    
    def vcg_post_stage(self, batch: dict):
        g = batch["g"]

        c_emb = g.nodes["c"].data["emb"]

        batch["output"] = torch.sigmoid(self.output_mlp(c_emb)).squeeze(-1)
        return batch

    def post_stage(self, batch: dict):
        task = self.config["task"]
        graph_type = self.config["graph_type"]

        if graph_type == "lcg":
            return self.lcg_post_stage(batch)
        elif graph_type == "vcg":
            return self.vcg_post_stage(batch)
        else:
            raise ValueError("Graph type not supported.")