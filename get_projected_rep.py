import argparse
from pathlib import Path
import numpy as np
import torch
from utils.loading_representations import loading_rep, loading_target
from utils.config import data_configs, model_configs
from utils.projection import stretch_along


def load_data(args):
    data_config = data_configs[args.data]
    train_rep = loading_rep(Path(data_config.data_dir, data_config.train_rep))
    valid_rep = loading_rep(Path(data_config.data_dir, data_config.valid_rep))
    return train_rep, valid_rep

def save_data(embeddings, path):
    np.save(path, embeddings)


def load_model(args):
    model_config = model_configs[args.probe_weights]
    return np.load(model_config.probe_dir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="projecting_rep_en",
        choices=list(data_configs.keys()),
    )
    parser.add_argument(
        "-p",
        "--probe_weights",
        type=str,
        choices=list(model_configs.keys()))
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data_config = data_configs[args.data]
    x_train, x_test = load_data(args)
    weights = load_model(args)
    projected_representation = stretch_along(torch.from_numpy(x_train), weights)
    save_data(projected_representation, data_config.train_rep[:-4]+'projected')
    projected_representation = stretch_along(torch.from_numpy(x_test), weights)
    save_data(projected_representation, data_config.valid_rep[:-4] + 'projected')


if __name__ == "__main__":
    main()


