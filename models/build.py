import torch


def build_model(configs):
    model_type = configs["model"]["type"]

    if model_type == "vanila_vae":
        from models.VAE.vanila_vae import VanillaVAE, Encoder, Decoder
        latent_dim = int(configs["model"]["latent_dim"])
        img_size = int(configs["model"]["img_size"])
        in_channels = int(configs["model"]['in_channels'])
        activation = configs["model"]['activation']
        encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, img_size=img_size)
        decoder = Decoder(out_channels=in_channels, latent_dim=latent_dim, img_size=img_size, activation=activation)
        model = VanillaVAE(encoder, decoder, latent_dim)
    
    elif model_type == 'vanila_gan':
        from models.GANs.vanila_gan import VanillaGAN, Generator, Discriminator
        in_channels = int(configs["model"]['in_channels'])
        latent_dim = int(configs["model"]["latent_dim"])
        img_size = int(configs["model"]["img_size"])
        
        generator = Generator(out_channels=in_channels, latent_dim=latent_dim, img_size=img_size)
        discriminator = Discriminator(in_channels=in_channels, img_size=img_size)
        
        model = VanillaGAN(generator, discriminator)
        
    elif model_type in ['ddpm', 'ddim']:
        from models.Diffusion.unet import UNet
        
        in_channels = int(configs["model"]['in_channels'])
        img_size = int(configs["model"]['img_size'])
        dim = int(configs["model"]['dim'])
        dim_mults = configs["model"]['dim_mults']
        num_res_blocks = int(configs["model"].get("num_res_blocks", 2))
        attn_layers = configs["model"].get("attn_layers", [])
        
        dropout = float(configs["model"].get("dropout", 0.0))
        model = UNet(
            dim=dim,
            dim_mults=dim_mults,
            attn_layers=attn_layers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            in_channels=in_channels,
            image_size=img_size
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    return model


def build_loss_function(configs):
    loss_type = configs["train"]['loss_fn']
    model_type = configs["model"]["type"]

    if model_type == 'vanila_vae':
        if loss_type == "mse":
            from models.VAE.vae_loss import vae_loss_function_mse
            loss_fn = vae_loss_function_mse
        elif loss_type == "bce":
            from models.VAE.vae_loss import vae_loss_function_bce
            loss_fn = vae_loss_function_bce
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
        
    elif model_type == 'vanila_gan':
        if loss_type == 'bce':
            from models.GANs.gan_loss import vanila_gan_loss
            loss_fn = vanila_gan_loss()
            
    elif model_type == 'ddpm':
        # Diffusion loss is implemented in the scheduler/model training loop.
        loss_fn = None
    else:
        raise ValueError(f"Unknown model: {model_type}")

    return loss_fn


def build_optimizer(model, configs):
    optim_type = configs["train"]['optimizer']
    learning_rate = float(configs["train"]['learning_rate'])
    weight_decay = float(configs["train"].get("weight_decay", 0))
    adam_betas = (float(configs["train"].get("beta1", 0.9)), float(configs["train"].get("beta2", 0.999)))
    
    if optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=adam_betas)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_type}")

    return optimizer


def build_diffusion_scheduler(configs, device):
    from models.Diffusion.diffusion_utils import linear_beta_schedule
    
    beta_start = float(configs["diffusion"]['beta_start'])
    beta_end = float(configs["diffusion"]['beta_end'])
    num_timesteps = int(configs["diffusion"]['num_timesteps'])

    betas = linear_beta_schedule(
        timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end
    )
    
    
    scheduler = configs['diffusion'].get('diffuser', 'ddpm_scheduler')
    scheduler_kwargs = {}
    
    if scheduler == 'ddpm_scheduler':
        from models.Diffusion.diffusion_utils import DDPMScheduler
        variance_type = configs["diffusion"].get("variance_type", "fixed_small")
        scheduler_kwargs["variance_type"] = variance_type
        DiffusionScheduler = DDPMScheduler
        
    elif scheduler == 'ddim_scheduler':
        from models.Diffusion.diffusion_utils import DDIMScheduler
        DiffusionScheduler = DDIMScheduler
    else:
        raise ValueError(f"Unknown diffusion scheduler: {scheduler}")

    diffusion = DiffusionScheduler(
        betas=betas,
        device=device,
        **scheduler_kwargs
    )
    
    return diffusion, num_timesteps

