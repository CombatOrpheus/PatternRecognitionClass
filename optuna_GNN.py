"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
from pathlib import Path

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from optuna.trial import TrialState

from src.datasets import get_reachability_dataset
from src.models import Petri_GCN

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial, num_features):
    return Petri_GCN(
        in_channels=num_features,
        hidden_features=trial.suggest_int("Hidden Features", 16, 128, 2),
        num_layers=trial.suggest_int("Number of GCN Layers", 2, 8),
        dropout=trial.suggest_float("Dropout", 0.0, 0.6),
        ################################################################
        # act=trial.suggest_categorical(                               #
        #     "Activation Function",                                   #
        #     ["ReLU", "LeakyReLU", "Sigmoid", "Softmin", "Softmax"]), #
        # norm=trial.suggest_categorical(                              #
        #     "Normalization Function",                                #
        #     ["GraphNorm", "LayerNorm", "BatchNorm"]),                #
        ################################################################
        readout_layers=trial.suggest_int("MLP Readout Layers", 2, 4)
        )


def get_data(trial):
    train_data = Path('Data/RandData_DS1_train_data.processed')
    test_data = Path('Data/RandData_DS1_test_data.processed')

    train_dataset = get_reachability_dataset(
        train_data,
        reduce_node_features=True,
        batch_size=trial.suggest_int("Batch size", 32, 128, 16))
    test_dataset = get_reachability_dataset(
        test_data,
        reduce_node_features=True,
        batch_size=128)
    return train_dataset, test_dataset


def objective(trial):
    train_loader, valid_loader = get_data(trial)
    # Generate the model.
    model = define_model(trial, train_loader.num_features).to(DEVICE)
    model = torch.compile(model, dynamic=True)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("Learning Rate", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data, data.y

            optimizer.zero_grad()
            output = torch.flatten(model(data))
            loss = F.l1_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data, data.y
                output = torch.flatten(model(data))

        error = F.l1_loss(output, data.y)

        trial.report(error, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return error


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=6000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
