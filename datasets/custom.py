import torch.utils.data as data
from torchvision import transforms
from utils.utils_images import get_image_paths, imread_uint


def get_transforms(img_size=None):
    return transforms.Compose([
        transforms.Resize(img_size) if img_size is not None else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class CustomDataset(data.Dataset):
    def __init__(self, configs):
        super().__init__()
        self.data_path = configs['dataset']['root']
        self.img_size = configs['model']['img_size']
        self.in_channels = configs['model']['in_channels']
        self.image_paths = get_image_paths(self.data_path)
        self.transform = get_transforms(self.img_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = imread_uint(image_path, self.in_channels)
        
        img = self.transform(img)
        
        return img, 0  # Dummy label