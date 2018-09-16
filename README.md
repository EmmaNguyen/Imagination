Pytorch implementation of Neural Scene and rendering
---
### About this project

#### Structure
```
.
├──docs
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
│       └──__init__.py
|── README.md
├── requirements.txt
├── test
│   ├── __init__.py
│   ├── test_data_transform.py
│   ├── test_gpu_setup.py
│   └── test_unittest.py
└── tools
    └── download_data.py
```

### Dataset

#### Download data - an easy way with one step

Here I provide a one-step preparation for you to (almost) immediately jump into the code. Only make sure that you have a good internet connection and around 2 GB storage (For a full data set, it is 23.68 GB)

```bash
$python tools/download_data.py --data_path=<path_to_your_data_directory>
```

#### Shepard Metzler 7 parts

The tiny preprocessed dataset described as above is extracted from Shepard Metzler with 7 parts composed of 7 randomly color cubes in three dimension grid with yaw (vertical axes) and pitch (transverse axes) followed by a position.

(IMAGE)

##### A comprehensive list of 7 datasets available for you to try

Checkout this ! [repository of DeepMind lab] (https://github.com/deepmind/gqn-datasets) for more information. This would require you to install some more packages such as `tensorflow` and `gsutils`. Following the instructions inside, you will be able to download 1.45 TB of raw data in total.  Since this is an implementation of `pytorch`, another extra step needed is converting those into tensor readable by our framework. Try this ! [open-source repository] (https://github.com/l3robot/gqn_datasets_translator)

### Requirements

##### 1. Using virtual environment with `Anaconda`

```bash
$bash scripts/download_and_setup_anaconda.sh
```

##### 2. Using `setuptools`

### Run the program
Simply as follows
```bash
$python main.py --data_path=<path_to_downloaded_data>
```

For further details of arguments, try with `--help`
```bash

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
