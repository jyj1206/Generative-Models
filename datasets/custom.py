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
        dataset_cfg = configs.get('dataset', {})

        self.data_path = dataset_cfg['root']
        self.img_size = dataset_cfg['img_size']
        self.in_channels = dataset_cfg['in_channels']

        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Custom dataset root does not exist: {self.data_path}")

        self.image_paths = get_image_paths(self.data_path)
        self.transform = get_transforms(self.img_size)

        self.use_text = False
        self.texts = None

        if configs.get('task') == 'latent_diffusion':
            self._enable_text_conditioning(dataset_cfg)

    def _enable_text_conditioning(self, dataset_cfg):
        text_cfg = dataset_cfg.get('text', None)
        if not isinstance(text_cfg, dict) or 'root' not in text_cfg:
            raise ValueError("For latent_diffusion with custom dataset, dataset.text.root is required.")

        text_root = text_cfg['root']
        if not os.path.isdir(text_root):
            raise FileNotFoundError(f"Caption root does not exist: {text_root}")

        txt_map = self._build_text_map(text_root)
        matched_image_paths, matched_texts = self._match_images_and_texts(txt_map)

        self.image_paths = matched_image_paths
        self.texts = matched_texts
        self.use_text = True

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f'No matched image-text pairs found.\n'
                f'image root: {self.data_path}\n'
                f'text root: {text_root}'
            )

    def _build_text_map(self, text_root):
        txt_files = glob.glob(os.path.join(text_root, '*.txt'))
        return {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in txt_files
        }

    def _match_images_and_texts(self, txt_map):
        matched_image_paths = []
        matched_texts = []

        for image_path in self.image_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            text_path = txt_map.get(image_name)

            if text_path is None:
                continue

            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            matched_image_paths.append(image_path)
            matched_texts.append(text)

        return matched_image_paths, matched_texts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = imread_uint(image_path, self.in_channels)
        img = self.transform(img)

        if self.use_text:
            return img, self.texts[index]

        return img, 0