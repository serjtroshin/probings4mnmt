from pathlib import Path
import numpy as np
import torch
from torch import nn
from utils.probing import Config, TorchRegressor, TorchClassifier, MLP
from utils.loading_representations import loading_rep, loading_target
from utils.config import DataConfig


class LinearWeightRegressor(TorchRegressor):
    def __init__(self, model: nn.Module, config: Config):
        super().__init__(model, config)

    @staticmethod
    def get_model(in_dim=768, out_dim=1):
        config = Config(
            training=Config.Training(lr=1e-3, weight_decay=1e-5, batch_size=32),
            model=Config.Model(
                in_dim=in_dim,
                out_dim=out_dim,
                n_hiddens=0,
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
    def get_model(in_dim=768, out_dim=3):
        config = Config(
            training=Config.Training(
                lr=1e-3,
                weight_decay=1e-5,
                batch_size=32,
                num_updates=10000,
                patience=10,
            ),
            model=Config.Model(
                in_dim=in_dim,
                out_dim=out_dim,
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


def load_data():
    data_config = DataConfig()
    rep = loading_rep(Path(data_config.data_dir, data_config.english_rep))
    label = loading_target(Path(data_config.data_dir, data_config.english_labels))
    return rep, label


def main():
    X, y = load_data()
    print("Unique labels:", np.unique(y))
    regressor = LinearWeightClassifier.get_model()
    regressor.fit(X, y)
    regressor.save_weights("models/linear_weight.pt")
    regressor.load_weights("models/linear_weight.pt")


if __name__ == "__main__":
    main()
