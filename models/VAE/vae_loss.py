import torch
import torch.nn as nn


def vae_loss_function_mse(outputs, inputs, beta=0.1):
    x_hat, mu, logvar = outputs

    recon_loss = nn.functional.mse_loss(x_hat, inputs, reduction='sum')

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss


def vae_loss_function_bce(outputs, inputs, beta=0.1):
    x_hat, mu, logvar = outputs
    loss_fn = nn.BCELoss(reduction='sum')
    
    recon_loss = loss_fn(x_hat, inputs)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss