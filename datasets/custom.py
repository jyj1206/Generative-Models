import os
import glob
import torch.utils.data as data
from torchvision import transforms
from utils.utils_images import get_image_paths, imread_uint


def get_transforms(img_size=None):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size) if img_size is not None else transforms.Identity(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class CustomDataset(data.Dataset):
    def __init__(self, configs):
        super().__init__()
        self.data_path = configs['dataset']['root']
        self.img_size = configs['dataset']['img_size']
        self.in_channels = configs['dataset']['in_channels']
        self.image_paths = get_image_paths(self.data_path)
        self.transform = get_transforms(self.img_size)

        self.use_text = False
        self.texts = None

        if configs['task'] == 'latent_diffusion':
            text_cfg = configs['dataset'].get('text', None)

            if isinstance(text_cfg, dict) and 'root' in text_cfg:
                text_root = text_cfg['root']
                txt_files = glob.glob(os.path.join(text_root, '*.txt'))
                txt_map = {
                    os.path.splitext(os.path.basename(path))[0]: path
                    for path in txt_files
                }

                matched_image_paths = []
                matched_texts = []

                for image_path in self.image_paths:
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    text_path = txt_map.get(image_name, None)

                    if text_path is None:
                        continue

                    with open(text_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()

                    matched_image_paths.append(image_path)
                    matched_texts.append(text)

                self.image_paths = matched_image_paths
                self.texts = matched_texts
                self.use_text = True

                if len(self.image_paths) == 0:
                    raise RuntimeError(
                        f'No matched image-text pairs found.\n'
                        f'image root: {self.data_path}\n'
                        f'text root: {text_root}'
                    )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = imread_uint(image_path, self.in_channels)
        img = self.transform(img)

        if self.use_text:
            return img, self.texts[index]

        return img, 0