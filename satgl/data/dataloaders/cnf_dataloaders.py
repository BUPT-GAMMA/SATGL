import dgl
import torch
import itertools
from dgl.dataloading import GraphCollator, GraphDataLoader

def satisfiability_collate_fn(data):
    return GraphCollator().collate([item for item in list(itertools.chain(*data))])

def maxsat_collate_fn(data):
    if (len(data) == 0):
        return None
    batched_data = {
        "g": GraphCollator().collate([elem["g"] for elem in data]),
        "label": torch.cat([torch.tensor(elem["label"]).float() for elem in data]),
        "info": GraphCollator().collate([elem["info"] for elem in data])
    }
    return batched_data

def unsat_core_collate_fn(data):
    if (len(data) == 0):
        return None
    batched_data = {
        "g": GraphCollator().collate([elem["g"] for elem in data]),
        "label": torch.cat([torch.tensor(elem["label"]).float() for elem in data]),
        "info": GraphCollator().collate([elem["info"] for elem in data])
    }
    return batched_data

def multitasks_collate_fn(data):
    if (len(data) == 0):
        return None
    
    # decode pair 
    data = [item for item in list(itertools.chain(*data))]
    
    # collate the label of each task
    num_tasks= len(data[0]["label"])
    labels = []
    for idx in range(num_tasks):
        label_elem = data[0]["label"][idx]
        # [list1, list2 ...]
        if isinstance(label_elem, list):
            labels.append(
                torch.cat([torch.tensor(elem["label"][idx]).float() for elem in data]),
            )
        # [num1, num2, ...]
        else:
            labels.append(
                GraphCollator().collate([torch.tensor(elem["label"][idx]).float() for elem in data]),
            )

    batched_data = {
        "g": GraphCollator().collate([elem["g"] for elem in data]),
        "label": labels,
        "info": GraphCollator().collate([elem["info"] for elem in data])
    }
    return batched_data

def SatistifiabilityDataLoader(dataset, batch_size, shuffle=False):
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=satisfiability_collate_fn,
    )


def MaxSATDataLoader(dataset, batch_size, shuffle=False):
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=maxsat_collate_fn,
    )

def UnsatCoreDataLoader(dataset, batch_size, shuffle=False):
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=unsat_core_collate_fn,
    )

def MultiTasksDataloader(dataset, batch_size, shuffle=False):
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=multitasks_collate_fn,
    )
