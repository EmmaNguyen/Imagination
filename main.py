"""
The main file to execute source code

Usage: $python main.py
"""

import argparse

import torch

from pytorch_deep_learning.utils.data_loader import ShepardMetzler
from pytorch_deep_learning.models.generative_query_network import GenerativeQueryNetwork
from pytorch_deep_learning.training import ModelTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="pytorch_deep_learning")

    parser.add_argument("--iterations", type=int, default=20000, \
        help="Number of batches to complete one epoch (Default: 20,000).")
    parser.add_argument("--batch_size", type=int, default=36, \
        help="Number of examples used for one batch.")
    parser.add_argument("--data_path", type=str, default="./data/train", \
        help="Path to directory of training data ")
    parser.add_argument("--model_path", type=str, default="./model", \
        help="Path to directory to save model with a timestamp.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    dataset = ShepardMetzler(data_path=args.data_path)
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=12)

    model_trainer = ModelTrainer(model, dataset, device)
    model_trainer.train(args.iterations)
