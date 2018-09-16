Pytorch implementation of Neural Scene and rendering
---
### About this project

#### Structure
```
├── docs
├── jupyter_notebook.ipynb
├── main.py
├── pytorch_deep_learning
│   ├── __init__.py
│   ├── models
│   │   ├── generative_architectures.py
│   │   ├── generative_query_network.py
│   │   ├── __init__.py
│   │   └── representation_architectures.py
│   ├── training.py
│   └── utils
│       ├── data_transform.py
│       ├── __init__.py
│       └── plots.py
├── README.md
├── requirements.txt
├── scripts
│   ├── deploy_slurm_jupyter_job_.sh
│   ├── download_and_setup_anaconda.sh
│   ├── generate_slurm_sbatch_history.sh
│   ├── retrieve_slurm_sbatch_command.sh
│   └── test_deploy_slurm_jupyter_job_.sh
├── setup.py
├── test
│   ├── __init__.py
│   ├── test_data_transform.py
│   ├── test_gpu_setup.py
│   ├── test_training.py
│   └── test_unittest.py
├── third_party
└── tools
    └── download_data.py
```

#### Contribution

### Dataset

#### Download data - an easy way with one step

#### A comprehensive list of all data

### Requirements

#### 1. Using virtual environment with `Ananconda`

#### 2. Using `setuptools`

### Run the program
Simply as follows
```
python main.py --data_path=<path_to_downloaded_data>
```

For further details of arguments, try with `--help`
```python

$python main.py --help                                                                                                                                                                                                                                                   [
usage: main.py [-h] [--iterations ITERATIONS] [--batch_size BATCH_SIZE]
               [--data_path DATA_PATH] [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --iterations ITERATIONS
                        A number of batches to complete one epoch (Default:
                        20,000).
  --batch_size BATCH_SIZE
                        A number of examples used for one batch. (Default: 36)
  --data_path DATA_PATH
                        A path to directory of training data
  --model_path MODEL_PATH
                        A path to directory to save a model with a timestamp.
```

### Future work

### References
