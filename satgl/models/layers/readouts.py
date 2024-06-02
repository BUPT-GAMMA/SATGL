import torch
import dgl
import torch.nn as nn

from typing import Union, List

from satgl.wrappers.model_wrappers.base_model_wrapper import ModelWrapper
from satgl.models.layers.mlp import MLP

class SatisfiabilityReadout(nn.Module):
    def __init__(
        self, 
        emb_size:int, 
        graph_type:str,
        num_fc:int = 3,
        **kwargs
    ) -> None:
        super(SatisfiabilityReadout, self).__init__()
        self.emb_size = emb_size
        self.graph_type = graph_type
        self.num_fc = num_fc
        
        if "use_core" in kwargs:
            self.use_core = True
            # todo: add satformer readout
        else:
            if graph_type in ["lcg", "lig"]:
                v_emb_size = emb_size * 2
            elif graph_type in ["vcg", "vig"]:
                v_emb_size = emb_size
            self.output_mlp = MLP(v_emb_size, emb_size, 1, num_layer=num_fc)
    
    def forward(self, batch: dict) -> dict:
        g = batch["g"]
        num_variables = batch["info"]["num_variables"]

        if self.graph_type in ["lcg", "lig"]:
            l_pos_emb = g.nodes["pos_l"].data["emb"]
            l_neg_emb = g.nodes["neg_l"].data["emb"]
            v_emb = torch.cat([l_pos_emb, l_neg_emb], dim=1)
        else:
            v_emb = g.nodes["v"].data["emb"]

        pool_emb = dgl.ops.segment_reduce(num_variables, v_emb, reducer="mean")
        batch["satisfiability"] = torch.sigmoid(self.output_mlp(pool_emb)).squeeze(-1)
        return batch

class MaxSATReadout(nn.Module):
    def __init__(
        self, 
        emb_size:int, 
        graph_type:str,
        num_fc:int = 3,
        **kwargs
    ) -> None:
        super(MaxSATReadout, self).__init__()
        self.emb_size = emb_size
        self.graph_type = graph_type
        self.num_fc = num_fc
        
        if graph_type in ["lcg", "lig"]:
            v_emb_size = emb_size * 2
        elif graph_type in ["vcg", "vig"]:
            v_emb_size = emb_size
        self.output_mlp = MLP(v_emb_size, emb_size, 1, num_layer=num_fc)
    
    def forward(self, batch: dict) -> dict:
        g = batch["g"]

        if self.graph_type in ["lcg", "lig"]:
            l_pos_emb = g.nodes["pos_l"].data["emb"]
            l_neg_emb = g.nodes["neg_l"].data["emb"]
            v_emb = torch.cat([l_pos_emb, l_neg_emb], dim=1)
        else:
            v_emb = g.nodes["v"].data["emb"]

        batch["maxsat"] = torch.sigmoid(self.output_mlp(v_emb)).squeeze(-1)
        return batch
    
class UnSATCoreReadout(nn.Module):
    def __init__(
        self, 
        emb_size:int, 
        graph_type:str,
        num_fc:int = 3,
        **kwargs
    ) -> None:
        super(UnSATCoreReadout, self).__init__()
        self.emb_size = emb_size
        self.graph_type = graph_type
        self.num_fc = num_fc
        
        self.output_mlp = MLP(emb_size, emb_size, 1, num_layer=num_fc)
    
    def forward(self, batch: dict) -> dict:
        g = batch["g"]

        c_emb = g.nodes["c"].data["emb"]

        batch["unsat_core"] = torch.sigmoid(self.output_mlp(c_emb)).squeeze(-1)
        return batch
    
class MultiTasksReadout(nn.Module):
    def __init__(
        self, 
        tasks:Union[str, List[str]],
        emb_size:int, 
        graph_type:str,
        num_fc:int = 3,
        **kwargs
    ) -> None:
        super(MultiTasksReadout, self).__init__()
        self.tasks = tasks if isinstance(tasks, list) else [tasks]
 
        self.readout_models = nn.ModuleList()
        for task in tasks:
            self.readout_models.append(task_to_module[task](emb_size, graph_type, num_fc))

    def forward(self, batch:dict) -> dict:
        batch["output"] = []
        for idx, task in enumerate(self.tasks):
            batch["output"].append(self.readout_models[idx](batch)[task])

        batch["g"].ndata.pop("emb")
        return batch
            

task_to_module = {
    "satisfiability": SatisfiabilityReadout,
    "maxsat": MaxSATReadout,
    "unsat_core": UnSATCoreReadout
}

def get_post_stage(
        tasks:Union[str, List[str]],
        emb_size:int, 
        graph_type:str,
        num_fc:int = 3,
        **kwargs
    ) -> nn.Module:
    return MultiTasksReadout(
        tasks,
        emb_size,
        graph_type,
        num_fc,
        **kwargs
    )
