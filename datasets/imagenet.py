from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

def get_transforms(img_size=128):
    return transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize(img_size, interpolation=InterpolationMode.BOX),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    
def get_imagenet_dataset(root, configs, train=True):
    img_size = configs['model']['img_size']
    transform = get_transforms(img_size)
    
    dataset = datasets.ImageNet(
        root=root,
        split='train' if train else 'val',
        download=True,
        transform=transform
    )
    
    return dataset

