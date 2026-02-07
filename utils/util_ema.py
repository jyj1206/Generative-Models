import torch

def ema_update(model, ema_model, decay):
    with torch.no_grad(): 
        model_params = dict(model.named_parameters())
        ema_params = dict(ema_model.named_parameters())

        for name in model_params.keys():
            ema_params[name].data.mul_(decay).add_(model_params[name].data, alpha=1 - decay)
            
        for name, buffer in model.named_buffers():
            ema_model.state_dict()[name].copy_(buffer)