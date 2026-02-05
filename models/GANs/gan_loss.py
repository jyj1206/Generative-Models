import torch.nn as nn


def vanila_gan_loss():
    return nn.BCEWithLogitsLoss()