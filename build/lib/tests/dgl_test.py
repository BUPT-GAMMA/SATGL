import dgl
import torch
import torch as th


import dgl.nn.pytorch as dglnn


if __name__ == "__main__":
    edges1 = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])
    edges2 = torch.tensor([0, 1]), torch.tensor([0, 1])
    edges3 = torch.tensor([0, 1]), torch.tensor([0, 1])

    gnn = dglnn.GraphConv(5, 5,  allow_zero_in_degree=True, weight=False, bias=False)

    g = dgl.heterograph({
        ('user', 'follows', 'user') : edges1,
        ('user', 'plays', 'game') : edges2,
        ('store', 'sells', 'game')  : edges3})
    
    conv = dglnn.HeteroGraphConv({
        'follows' : gnn,
        'plays' : gnn,
        'sells' : gnn},
        aggregate='sum')
        
    h1 = {
        'user' : th.ones((g.number_of_nodes('user'), 5)),
        'game' : th.zeros((g.number_of_nodes('game'), 5)),
        'store' : th.ones((g.number_of_nodes('store'), 5))
    }
    h2 = conv(g, h1)
    print(h2["user"])
    print(h2["game"])

