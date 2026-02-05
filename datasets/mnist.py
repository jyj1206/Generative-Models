from torchvision import datasets, transforms


def get_transforms_bce():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    
def get_transforms_tanh():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    
def get_mnist_dataset(root, train=True, transform_type="bce"):
    if transform_type == "bce":
        transform = get_transforms_bce()
    elif transform_type == "tanh":
        transform = get_transforms_tanh()
    else:
        raise ValueError("Invalid transform type. Use 'bce' or 'tanh'.")
    
    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    
    return dataset