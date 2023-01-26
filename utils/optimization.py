import argparse
import logging
import torch
from torch import nn
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, mean_absolute_error
from utils.probing import Config, TorchRegressor, TorchClassifier, MLP


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


def TorchLinearRegressor(args: argparse.Namespace):
    # 1. Define an objective function to be maximized.
    class Model:
        def __init__(self):
            pass

        def fit(self, X_train, y_train):
            n_features = X_train.shape[-1]

            def objective(trial: optuna.Trial):

                # 2. Suggest values of the hyperparameters using a trial object.
                lr = trial.suggest_float("lr", high=0.1, low=0.0001, log=True)
                weight_decay = trial.suggest_float(
                    "weight_decay", high=0.1, low=0.00001, log=True
                )

                config = Config(
                    verbose=False,
                    cuda_if_avail=True,
                    model=Config.Model(in_dim=n_features, n_hiddens=0, dropout=0),
                    training=Config.Training(
                        lr=lr,
                        weight_decay=weight_decay,
                        batch_size=args.batch_size,
                        num_updates=args.num_updates,
                        patience=args.patience,
                    ),
                )
                model = LinearWeightRegressor(
                    config=config, model=MLP(args=config.model)
                )
                train_losses, val_losses, X_val, y_val = model.fit(
                    X_train, y_train.astype("float32"), return_val=True
                )

                objective = mean_absolute_error(model.predict(X_val), y_val)
                return objective

            # 3. Create a study object and optimize the objective function.
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.n_trials)
            best_params = study.best_params

            config = Config(
                verbose=False,
                cuda_if_avail=True,
                model=Config.Model(in_dim=n_features, n_hiddens=0, dropout=0),
                training=Config.Training(
                    lr=best_params["lr"],
                    weight_decay=best_params["weight_decay"],
                    batch_size=args.batch_size,
                    num_updates=args.num_updates,
                    patience=args.patience,
                ),
            )
            model = LinearWeightRegressor(config=config, model=MLP(args=config.model))
            model.fit(X_train, y_train.astype("float32"))
            self.model = model

        def predict(self, X_train):
            return np.nan_to_num(self.model.predict(X_train))

        def save_weights(self, path):
            self.model.save_weights(path)

        def load_weights(self, path):
            self.model.load_weights(path)

    return Model()


def TorchLinearClassifier(args: argparse.Namespace):
    # 1. Define an objective function to be maximized.
    class Model:
        def __init__(self):
            pass

        def fit(self, X_train, y_train):
            print(
                f"MLP classifier got: X_train {X_train.shape}, y_train: {y_train.shape}"
            )
            n_features = X_train.shape[-1]
            n_out = y_train.max() + 1

            def objective(trial: optuna.Trial):

                # 2. Suggest values of the hyperparameters using a trial object.
                lr = trial.suggest_float("lr", high=0.1, low=0.0001, log=True)
                weight_decay = trial.suggest_float(
                    "weight_decay", high=0.1, low=0.00001, log=True
                )

                config = Config(
                    verbose=False,
                    cuda_if_avail=True,
                    model=Config.Model(
                        in_dim=n_features, out_dim=n_out, n_hiddens=0, dropout=0
                    ),
                    training=Config.Training(
                        lr=lr,
                        weight_decay=weight_decay,
                        batch_size=args.batch_size,
                        num_updates=args.num_updates,
                        patience=args.patience,
                    ),
                )
                model = LinearWeightClassifier(config=config, model=MLP(args=config.model))
                train_losses, val_losses, X_val, y_val = model.fit(
                    X_train, y_train, return_val=True
                )

                objective = accuracy_score(model.predict(X_val), y_val)
                return objective

            # 3. Create a study object and optimize the objective function.
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.n_trials)
            best_params = study.best_params

            config = Config(
                verbose=False,
                cuda_if_avail=True,
                model=Config.Model(
                    in_dim=n_features, out_dim=n_out, n_hiddens=0, dropout=0
                ),
                training=Config.Training(
                    lr=best_params["lr"],
                    weight_decay=best_params["weight_decay"],
                    batch_size=args.batch_size,
                    num_updates=args.num_updates,
                    patience=args.patience,
                ),
            )
            model = LinearWeightClassifier(config=config, model=MLP(args=config.model))
            model.fit(X_train, y_train)
            self.model = model

        def predict(self, X_train):
            pred = np.nan_to_num(self.model.predict(X_train))
            print(f"pred: {pred.shape}")
            return pred
            # return pred.squeeze(-1)

        def save_weights(self, path):
            self.model.save_weights(path)

        def load_weights(self, path):
            self.model.load_weights(path)

    return Model()
