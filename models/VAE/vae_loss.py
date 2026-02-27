import torch
import torch.nn as nn


def vae_loss_function_mse(outputs, inputs, beta=0.1):
    x_hat, mu, logvar = outputs
    batch_size = inputs.size(0)

    recon_loss = nn.functional.mse_loss(x_hat, inputs, reduction='sum') / batch_size

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    return recon_loss + beta * kl_loss