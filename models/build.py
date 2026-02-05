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
        discriminator = Discriminator(in_channels=in_channels)
        
        model = VanillaGAN(generator, discriminator)
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