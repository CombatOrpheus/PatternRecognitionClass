from pathlib import Path

import optuna
import torch
import torch.optim as optim
import torch.utils.data
from optuna.trial import TrialState
from tqdm import trange

from src.datasets import get_average_tokens_dataset
from src.models import Petri_GCN

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EPOCHS = 100
MAE = True
REDUCE_FEATURES = False


def define_model(trial, num_features):
    return Petri_GCN(
        in_channels=num_features,
        hidden_channels=trial.suggest_int("Hidden Features", 2, 16, step=2),
        num_layers=trial.suggest_int("Number of GCN Layers", 2, 20),
        readout_layers=trial.suggest_int("MLP Readout Layers", 2, 5),
        mae=MAE
        )


def train_model(model, optimizer, train_loader):
    model.train()
    for graph in train_loader:
        data, target = graph, graph.y
        optimizer.zero_grad()
        output = torch.flatten(model(data))
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()
    return loss


def eval_model(model, valid_loader):
    model.eval()
    actual = torch.tensor([graph.y for graph in valid_loader.dataset])
    pred = [model(graph).tolist() for graph in valid_loader]
    pred = torch.tensor(pred)
    pred = torch.flatten(pred)

    return model.loss(pred, actual)


def get_data(trial):
    train_data = Path('Data/RandData_DS1_train_data.processed')
    test_data = Path('Data/RandData_DS1_test_data.processed')

    batch_size = trial.suggest_int("Batch Size", 16, 128, step=4)

    train_dataset = get_average_tokens_dataset(
        train_data,
        reduce_features=REDUCE_FEATURES,
        batch_size=batch_size)

    test_dataset = get_average_tokens_dataset(
        test_data,
        reduce_features=REDUCE_FEATURES,
        batch_size=batch_size)
    return train_dataset, test_dataset


def objective(trial):
    train_loader, test_loader = get_data(trial)
    # Generate the model.
    model = define_model(trial, train_loader.num_features).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("Learning Rate", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    for epoch in trange(EPOCHS, desc='Trial loop', leave=False):
        loss = train_model(model, optimizer, train_loader)
        trial.report(loss, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return loss


def detailed_objective(trial):
    train_loader, test_loader = get_data(trial)
    model = define_model(trial, train_loader.num_features).to(DEVICE)

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("Learning Rate", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    for epoch in trange(EPOCHS, desc='Trial loop', leave=False):
        loss = train_model(model, optimizer, train_loader)
        trial.report(loss, epoch)

    results = {}
    datasets = Path('Data').glob('*all*')
    for dataset in datasets:
        data = get_average_tokens_dataset(dataset, reduce_features=REDUCE_FEATURES)
        actual = torch.tensor(([graph.y for graph in data]))
        pred = [model(graph).tolist() for graph in data]
        pred = torch.tensor(pred)
        pred = torch.flatten(pred)

        results['dataset'] = model.loss(pred, actual)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=6000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"""Study statistics:
                Number of finished trials: ", {len(study.trials)}
                Number of pruned trials: ", {len(pruned_trials)}
                Number of complete trials: ", {len(complete_trials)}""")

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    print("Evaluating best model...")
    detailed_objective(best_trial)
