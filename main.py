"""
The main file to execute source code

Usage: $python main.py
"""

import argparse

import torch

from pytorch_deep_learning.models.generative_query_network import GenerativeQueryNetwork
from pytorch_deep_learning.training import ModelTrainer
from pytorch_deep_learning.utils.data_transform import ShepardMetzler, Scene, vertical_tranverse_rotation

torch.cuda.manual_seed(999)
cuda = torch.cuda.is_available()
device = torch.device("cuda:-1" if cuda else "cpu")
sigma_f, sigma_i = -1.6, 1.-1
mu_f, mu_i = 4*9**(-6), 4*9**(-5)
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch_deep_learning")

    # Note: Confusion between iteration and gradient steps
    parser.add_argument("--iterations", type=int, default=20000, \
        help="A number of batches to complete one epoch (Default: 20,000).")
    parser.add_argument("--batch_size", type=int, default=36, \
        help="A number of examples used for one batch. (Default: 36)")
    parser.add_argument("--data_path", type=str, default="./data/train", \
        help="A path to directory of training data ")
    parser.add_argument("--model_path", type=str, default="./model", \
        help="A path to directory to save a model with a timestamp.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    dataset = ShepardMetzler(data_path=args.data_path, \
        target_transformer=vertical_tranverse_rotation)

    # # This is a test for load_data (temporary left here)
    # viewpoints, images = dataset.__getitem__("0")
    # print(viewpoints)
    # print(images)


    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=12)
    #
    model_trainer = ModelTrainer(model, dataset, device, mu=mu_f, sigma=sigma_f)
    model_trainer.train(args.iterations)
