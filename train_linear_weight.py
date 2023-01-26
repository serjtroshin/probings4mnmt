import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from utils.probing import Config, TorchRegressor, TorchClassifier, MLP
from utils.loading_representations import loading_rep, loading_target
from utils.config import data_configs


class LinearWeightRegressor(TorchRegressor):
    def __init__(self, model: nn.Module, config: Config):
        super().__init__(model, config)

    @staticmethod
    def get_model(args):
        config = Config(
            training=Config.Training(
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                num_updates=args.num_updates,
                patience=args.patience,
            ),
            model=Config.Model(
                in_dim=args.in_dim,
                out_dim=args.out_dim,
                n_hiddens=0,  # Linear Model
            ),
        )
        model = MLP(config.model)
        regressor = LinearWeightRegressor(model, config)
        return regressor

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))


class LinearWeightClassifier(TorchClassifier):
    def __init__(self, model: nn.Module, config: Config):
        super().__init__(model, config)

    @staticmethod
    def get_model(args: argparse.Namespace):
        config = Config(
            training=Config.Training(
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                num_updates=args.num_updates,
                patience=args.patience,
            ),
            model=Config.Model(
                in_dim=args.in_dim,
                out_dim=args.out_dim,
                n_hiddens=0,  # Linear Model
            ),
        )
        model = MLP(config.model)
        regressor = LinearWeightClassifier(model, config)
        return regressor

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="bert_english",
        choices=list(data_configs.keys()),
    )
    # Training args
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_updates", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=10)
    # Model args
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--out_dim", type=int, default=3)
    parser.add_argument(
        "--model_type",
        type=str,
        default="regressor",
        choices=["regressor", "classifier"],
    )
    args = parser.parse_args()
    return args


def load_data(args):
    data_config = data_configs[args.data]
    rep = loading_rep(Path(data_config.data_dir, data_config.english_rep))
    label = loading_target(Path(data_config.data_dir, data_config.english_labels))
    return rep, label


def main():
    args = get_args()
    print(args)
    X, y = load_data(args)
    print("Unique labels:", np.unique(y))
    if args.model_type == "regressor":
        model = LinearWeightRegressor.get_model(args)
    else:
        model = LinearWeightClassifier.get_model(args)
    model.fit(X, y)
    save_to = f"models/linear_model_{args.data}.pt"
    model.save_weights(save_to)
    model.load_weights(save_to)


if __name__ == "__main__":
    main()
