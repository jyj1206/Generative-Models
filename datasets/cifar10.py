from torchvision import datasets, transforms


def get_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    

def get_cifar10_dataset(root, train=True):
    transform = get_transforms() 
    
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    
    return dataset