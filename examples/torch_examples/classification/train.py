"""
Simple PyTorch pipeline with training
classification model on the Fashion-MNIST dataset.

Reference:
https://github.com/bstollnitz/fashion-mnist-pytorch/blob/main/fashion-mnist-pytorch/src/main.py
"""

from pathlib import Path
from typing import Tuple, Dict

from time import time
import numpy as np
import torch
import json
import os

from PIL import Image
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from fvcore.nn import FlopCountAnalysis

from evolly import compute_fitness

from create_model import my_model

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}

DATA_DIRPATH = 'fashion-mnist/data'
IMAGE_FILEPATH = 'fashion-mnist/src/predict-image.png'


def main() -> None:

    from cfg import cfg

    cfg.model.name = '0000_00001'

    config_path = None          # path to config file
    if config_path:
        cfg.merge_from_file(config_path)

    # Assign which accelerator to use during training
    cfg.train.accelerators = ['cuda:0' if torch.cuda.is_available() else 'cpu']
    cfg.train.accelerator_type = 'GPU'

    train_wrapper(cfg)


def train_wrapper(cfg) -> None:

    # Here weights of the last epoch will be returned,
    # but it's better to return weights of the "best" epoch
    model, meta_data = train(cfg)

    # Compute fitness value
    meta_data['fitness'] = compute_fitness(
        val_metrics=meta_data['val_metric'],
        target_params=cfg.search.target,
        model_params=meta_data['parameters'],
        w=cfg.search.w,
        metric_op=cfg.val.metric_op
    )

    # Save trained model to file
    cfg.model.name += f'_{meta_data["fitness"]:.5f}'
    Path(cfg.train.save_dir).mkdir(exist_ok=True)
    path = Path(cfg.train.save_dir, cfg.model.name)
    torch.save(model.state_dict(), path)

    # Save metadata json
    metadata_path = os.path.join(cfg.train.save_dir, f'{cfg.model.name}_meta.json')
    save_json(metadata_path, meta_data)


def train(cfg) -> Tuple[nn.Module, Dict]:
    """Trains the model for a number of epochs, and saves it."""
    learning_rate = 0.1
    batch_size = 64

    device = cfg.train.accelerators[0]

    meta_data = {'train_loss': [], 'val_metric': [], 'config': cfg}

    start_time = time()

    (train_dataloader, test_dataloader) = _get_data(batch_size)

    # Build model from genotype
    model = my_model(cfg)

    img_inputs, _ = next(iter(train_dataloader))
    meta_data['parameters'] = int(sum(p.numel() for p in model.parameters()))
    meta_data['flops'] = int(FlopCountAnalysis(model, img_inputs).total())

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print('\n***Training***')
    for epoch in range(cfg.train.epochs):

        print(f'\nEpoch {epoch + 1}\n-------------------------------')
        (train_loss, train_accuracy) = _fit(device, train_dataloader, model,
                                            loss_fn, optimizer)
        print(f'Train loss: {train_loss:>8f}, ' +
              f'train accuracy: {train_accuracy * 100:>0.1f}%')

        meta_data['train_loss'].append(train_loss)

        (test_loss, test_accuracy) = _evaluate(device, test_dataloader, model,
                                               loss_fn)
        print(f'Test loss: {test_loss:>8f}, ' +
              f'test accuracy: {test_accuracy * 100:>0.1f}%')

        meta_data['val_metric'].append(test_accuracy)

    meta_data['training_time'] = float(time() - start_time)

    return model, meta_data


def _get_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Downloads Fashion MNIST data, and returns two DataLoader objects
    wrapping test and training data."""
    training_data = datasets.FashionMNIST(
        root=DATA_DIRPATH,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root=DATA_DIRPATH,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def _fit(device: str, dataloader: DataLoader, model: nn.Module,
         loss_fn: CrossEntropyLoss,
         optimizer: Optimizer) -> Tuple[float, float]:
    """Trains the given model for a single epoch."""
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    # Used for printing only.
    batch_count = len(dataloader)
    print_every = 100

    model.to(device)
    model.train()

    for batch_index, (x, y) in enumerate(dataloader):
        x = x.float().to(device)
        y = y.long().to(device)

        (y_prime, loss) = _fit_one_batch(x, y, model, loss_fn, optimizer)

        correct_item_count += (y_prime.argmax(1) == y).sum().item()
        loss_sum += loss.item()
        item_count += len(x)

        # Printing progress.
        if ((batch_index + 1) % print_every == 0) or ((batch_index + 1)
                                                      == batch_count):
            accuracy = correct_item_count / item_count
            average_loss = loss_sum / item_count
            print(f'[Batch {batch_index + 1:>3d} - {item_count:>5d} items] ' +
                  f'loss: {average_loss:>7f}, ' +
                  f'accuracy: {accuracy*100:>0.1f}%')

    average_loss = loss_sum / item_count
    accuracy = correct_item_count / item_count

    return average_loss, accuracy


def _fit_one_batch(x: torch.Tensor, y: torch.Tensor, model: nn.Module,
                   loss_fn: CrossEntropyLoss,
                   optimizer: Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trains a single minibatch (backpropagation algorithm)."""
    y_prime = model(x)
    loss = loss_fn(y_prime, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return y_prime, loss


def _evaluate(device: str, dataloader: DataLoader, model: nn.Module,
              loss_fn: CrossEntropyLoss) -> Tuple[float, float]:
    """Evaluates the given model for the whole dataset once."""
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.float().to(device)
            y = y.long().to(device)

            (y_prime, loss) = _evaluate_one_batch(x, y, model, loss_fn)

            correct_item_count += (y_prime.argmax(1) == y).sum().item()
            loss_sum += loss.item()
            item_count += len(x)

        average_loss = loss_sum / item_count
        accuracy = correct_item_count / item_count

    return average_loss, accuracy


def _evaluate_one_batch(
        x: torch.tensor, y: torch.tensor, model: nn.Module,
        loss_fn: CrossEntropyLoss) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluates a single minibatch."""
    with torch.no_grad():
        y_prime = model(x)
        loss = loss_fn(y_prime, y)

    return y_prime, loss


def _predict(model: nn.Module, x: torch.Tensor, device: str) -> np.ndarray:
    """Makes a prediction for input x."""
    model.to(device)
    model.eval()

    x = torch.from_numpy(x).float().to(device)

    with torch.no_grad():
        y_prime = model(x)
        probabilities = nn.functional.softmax(y_prime, dim=1)
        predicted_indices = probabilities.argmax(1)
    return predicted_indices.cpu().numpy()


def inference_phase(device: str, cfg):
    """Makes a prediction for a local image."""
    print('\n***Predicting***')

    model = my_model(cfg)
    path = Path(cfg.train.save_dir, 'weights.pth')
    model.load_state_dict(torch.load(path))

    with Image.open(IMAGE_FILEPATH) as image:
        x = np.asarray(image).reshape((-1, 28, 28)) / 255.0

    predicted_index = _predict(model, x, device)[0]
    predicted_class = labels_map[predicted_index]

    print(f'Predicted class: {predicted_class}')


def save_json(path, output_dict):
    with open(path, "w") as j:
        json.dump(output_dict, j, indent=2)


if __name__ == '__main__':
    main()
