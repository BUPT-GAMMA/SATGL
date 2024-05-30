## Install

### create python 3.9 environment

```
conda create -n python=3.9
```

### install python packages

```
pip install -r requirements.txt
```

### install dgl following https://www.dgl.ai/pages/start.html


### install satgl

```
pip install ./dist/satgl-0.1-py3-none-any.whl
```

## demo example

run example in the test directory

### label_gen

```
python ./label_gen --cnf_dir YOUR_CNF_DIR
```

### a simple example for train

the dataset should be in the format like this

```
root_dir
├── train
├── valid
├── test
└── label
```

```
import os
import sys
import torch
import torch.nn as nn

from satgl.run_experiment import run_experiment
from satgl.config.config import Config

if __name__ == "__main__":
    config_files = ["./test_yaml/neurosat.yaml"]
    parameter_dict = {
        "task": "satisfiability",
        "model": "neurosat",
        "dataset_path": "../benchmarks/sr",
    }
    config = Config(config_files, parameter_dict)
    run_experiment(config)

```

then run the example 

```
python ./neurosat_test.py --dataset_path=YOUR_DATASET_PATH
```