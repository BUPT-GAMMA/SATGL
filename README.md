## Install

### install python 3.9

```
conda create -n python=3.9
```

### install python packages

```
pip install -r requirements.txt
```

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

### train

the dataset should be in the format like this

```
root_dir
├── train
├── valid
├── test
└── label
```



```
python ./neurosat_test.py --dataset_path=YOUR_DATASET_PATH
```