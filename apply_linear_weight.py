import argparse
from pathlib import Path
import numpy as np
import torch

from utils.loading_representations import loading_rep, loading_target
from utils.config import data_configs
from utils.optimization import (
    LinearWeightRegressor,
    LinearWeightClassifier,
)


def load_data(args):
    data_config = data_configs[args.eval_data]
    valid_rep = loading_rep(Path(data_config.data_dir, data_config.valid_rep))
    valid_label = loading_target(Path(data_config.data_dir, data_config.valid_labels))
    return valid_rep, valid_label


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--train_data",
        type=str,
        default="bert_english",
        choices=list(data_configs.keys()),
    )
    parser.add_argument(
        "-e",
        "--eval_data",
        type=str,
        default="bert_greek",
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
        default="classifier",
        choices=["regressor", "classifier"],
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    X_test, y_test = load_data(args)

    print("Unique labels:", np.unique(y_test))
    if args.model_type == "regressor":
        model = LinearWeightRegressor.get_model(args)
        metric = (
            lambda y, y_pred: -torch.nn.functional.mse_loss(y, y_pred).mean().item()
        )
        metric_name = "mse"
    else:
        model = LinearWeightClassifier.get_model(args)
        metric = lambda y, y_pred: np.mean(y == y_pred)
        metric_name = "accuracy"
    save_to = f"models/linear_model_{args.train_data}.npy"
    model.load_weights(save_to)
    y_pred = model.predict(X_test)
    print(f"Test metric ({metric_name}):", metric(y_test, y_pred))


if __name__ == "__main__":
    main()
