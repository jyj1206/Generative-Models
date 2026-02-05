from datasets.cifar10 import get_cifar10_dataset
from datasets.mnist import get_mnist_dataset

def build_dataset(cfg):
    name = cfg["dataset"]["name"]
    root = cfg["dataset"]["root"]

    if name == "cifar10":
        train_dataset = get_cifar10_dataset(root, train=True)
        test_dataset = get_cifar10_dataset(root, train=False)

    elif name == "mnist":
        train_dataset = get_mnist_dataset(root, train=True)
        test_dataset = get_mnist_dataset(root, train=False)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_dataset, test_dataset
