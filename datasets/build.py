
def build_dataset(configs):
    task = configs['task']
    name = configs["dataset"]["name"]
    root = configs["dataset"]["root"]

    if name == "cifar10":
        from datasets.cifar10 import get_cifar10_dataset
        train_dataset = get_cifar10_dataset(root, train=True)
        if task == 'vae':
            test_dataset = get_cifar10_dataset(root, train=False)
        else:
            test_dataset = None

    elif name == "mnist":
        from datasets.mnist import get_mnist_dataset
        
        train_dataset = get_mnist_dataset(root, train=True)
        
        if task == 'vae':
            test_dataset = get_mnist_dataset(root, train=False)
        else:
            test_dataset = None

    elif name == "imagenet":
        from datasets.imagenet import get_imagenet_dataset
        train_dataset = get_imagenet_dataset(root, train=True, configs=configs)
        test_dataset = None  # TODO: Implement test dataset for ImageNet

    elif name == "custom":
        from datasets.custom import CustomDataset
        train_dataset = CustomDataset(configs)
        test_dataset = None # TODO: Implement test dataset for custom dataset        

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_dataset, test_dataset
