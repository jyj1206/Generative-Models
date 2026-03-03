from torchvision import datasets, transforms


def get_transforms(img_size=32):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    

def get_cifar10_dataset(root, configs, train=True):
    img_size = configs['dataset']['img_size']
    transform = get_transforms(img_size)
    
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    
    return dataset