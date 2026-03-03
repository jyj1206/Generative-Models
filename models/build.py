import torch


class HingeAdversarialLoss:
    def generator_loss(self, logits_fake):
        return -torch.mean(logits_fake)

    def discriminator_loss(self, logits_real, logits_fake):
        loss_real = torch.relu(1.0 - logits_real).mean()
        loss_fake = torch.relu(1.0 + logits_fake).mean()
        return 0.5 * (loss_real + loss_fake)


def build_model(configs):
    task = configs["task"]
    model_type = configs["model"]["type"]

    if task == 'vae':
        if model_type == "vanila_vae":
            from models.VAE.nets.vanila_vae import VanillaVAE
            from models.VAE.nets.encoder.vanila_encoder import Encoder
            from models.VAE.nets.decoder.vanila_decoder import Decoder
            encoder = Encoder(
                in_channels=int(configs["model"]["in_channels"]),
                latent_dim=int(configs["model"]["latent_dim"]),
                img_size=int(configs["model"]["img_size"]),
            )
            decoder = Decoder(
                out_channels=int(configs["model"]["in_channels"]),
                latent_dim=int(configs["model"]["latent_dim"]),
                img_size=int(configs["model"]["img_size"]),
            )
            model = VanillaVAE(encoder, decoder, int(configs["model"]["latent_dim"]))
        elif model_type == "vqvae":
            reg_type = configs["model"].get("reg_type", "vq")
            disc_init_weights = bool(configs["model"].get("disc_init_weights", False))
            from models.VAE.nets.vqvae import VQVAE
            from models.VAE.nets.vae import VAE
            from models.others.lpips import LPIPS
            from models.GAN.nets.discriminator.patch_gan_discriminator import PatchGANDiscriminator
            
            if reg_type == "vq":
                autoencoder = VQVAE(
                    in_channels=int(configs["model"]['in_channels']),
                    z_channels=int(configs["model"]["latent_dim"]),
                    codebook_size=int(configs["model"]["num_embeddings"]),
                )
            elif reg_type == "kl":
                autoencoder = VAE(
                    in_channels=int(configs["model"]['in_channels']),
                    z_channels=int(configs["model"]["latent_dim"]),
                )
            else:
                raise ValueError(f"Unknown regularization type for vqvae model: {reg_type}")

            model = {
                'vqvae': autoencoder,
                'lpips': LPIPS(),
                'discriminator': PatchGANDiscriminator(
                    in_channels=int(configs["model"]['in_channels']),
                    init_weights=disc_init_weights,
                )
            }
        else:
            raise ValueError(f"Unknown VAE model: {model_type}")
        
    elif task == 'gan':
        if model_type == 'vanila_gan':
            from models.GAN.nets.vanila_gan import VanillaGAN
            from models.GAN.nets.generator.vanila_generator import Generator
            from models.GAN.nets.discriminator.vanila_discriminator import Discriminator
            generator = Generator(
                out_channels=int(configs["model"]["in_channels"]),
                latent_dim=int(configs["model"]["latent_dim"]),
                img_size=int(configs["model"]["img_size"]),
            )
            discriminator = Discriminator(
                in_channels=int(configs["model"]["in_channels"]),
                img_size=int(configs["model"]["img_size"]),
            )
            
            model = VanillaGAN(generator, discriminator)
        else:
            raise ValueError(f"Unknown GAN model: {model_type}")

    elif task == 'diffusion':   
        if model_type in ['ddpm', 'ddim']:
            from models.Diffusion.nets.unet import UNet

            model = UNet(
                dim=int(configs["model"]["dim"]),
                dim_mults=configs["model"]["dim_mults"],
                attn_layers=configs["model"].get("attn_layers", []),
                num_res_blocks=int(configs["model"].get("num_res_blocks", 2)),
                dropout=float(configs["model"].get("dropout", 0.0)),
                in_channels=int(configs["model"]["in_channels"]),
                image_size=int(configs["model"]["img_size"]),
                num_classes=configs.get("dataset", {}).get("num_classes", None),
                use_biggan_resample=bool(configs["model"].get("use_biggan_resample", False)),
            )
        else:
             raise ValueError(f"Unknown diffusion model: {model_type}")
    
    elif task == 'stable_diffusion':
        if model_type in ['ddpm', 'ddim']:
            from models.Diffusion.nets.unet import UNet
            
            reg_type = configs["first_stage"].get("reg_type", "vq")
            first_stage_cfg = configs["first_stage"]
            
            if reg_type == "vq":
                from models.VAE.nets.vqvae import VQVAEInterface
                latent_dim = int(first_stage_cfg.get("latent_dim", configs["model"]["in_channels"]))
                first_stage = VQVAEInterface(
                    embed_dim=latent_dim,
                    in_channels=int(first_stage_cfg.get("in_channels", 3)),
                    z_channels=latent_dim,
                    codebook_size=int(first_stage_cfg.get("num_embeddings", 8192)),
                )
            elif reg_type == "kl":
                from models.VAE.nets.vae import VAE
                first_stage = VAE(
                    in_channels=int(first_stage_cfg.get('in_channels', 3)),
                    z_channels=int(first_stage_cfg["latent_dim"]),
                )
            else:
                raise ValueError(f"Unknown regularization type for first stage model: {reg_type}")
            
            second_stage =  UNet(
                    dim=int(configs["model"]["dim"]),
                    dim_mults=configs["model"]["dim_mults"],
                    attn_layers=configs["model"].get("attn_layers", []),
                    num_res_blocks=int(configs["model"].get("num_res_blocks", 2)),
                    dropout=float(configs["model"].get("dropout", 0.0)),
                    in_channels=int(configs["model"]["in_channels"]),
                    image_size=int(configs["model"]["img_size"]),
                    num_classes=configs.get("dataset", {}).get("num_classes", None),
                    use_biggan_resample=bool(configs["model"].get("use_biggan_resample", False)),
                )
            
            model = {
                'first_stage': first_stage,
                'second_stage': second_stage
            }
        else:
             raise ValueError(f"Unknown stable diffusion model: {model_type}")
    
         
    else:
        raise ValueError(f"Unknown task: {task}")

    return model


def build_loss_function(configs):
    task = configs["task"]
    loss_type = configs["train"].get('loss_fn', None)
    model_type = configs["model"]["type"]
    
    if task == 'vae':
        if model_type == 'vanila_vae':
            beta = float(configs["train"].get("beta", 0.1))
            warmup_epochs = int(configs["train"].get("kl_warmup_epochs", 0))
        
            if loss_type == 'mse':
                from models.VAE.vae_loss import vanila_vae_loss_function_mse
                def loss_fn(outputs, inputs, epoch=None):
                    current_beta = beta
                    if epoch is not None and warmup_epochs > 0:
                        progress = min(1.0, float(epoch) / float(warmup_epochs))
                        current_beta = beta * progress
                    return vanila_vae_loss_function_mse(outputs, inputs, beta=current_beta)
            else:
                raise ValueError(f"Unknown loss function: {loss_type}")
            
        elif model_type == 'vqvae':
            loss_fn = {
                'recon_criterion': torch.nn.MSELoss(),
                'discriminator_criterion': HingeAdversarialLoss(),
            }   
        else:
            raise ValueError(f"Unknown VAE model: {model_type}")
             
        
    elif task == 'gan':    
        if model_type == 'vanila_gan':
            if loss_type == 'bce':
                from models.GAN.gan_loss import vanila_gan_loss_bce
                loss_fn = vanila_gan_loss_bce()
            else:
                raise ValueError(f"Unknown loss function: {loss_type}")
        else:
            raise ValueError(f"Unknown GAN model: {model_type}")
                
    elif task == 'diffusion':
        if model_type in ['ddpm', 'ddim']:
            # Diffusion loss is implemented in the scheduler/model training loop.
            loss_fn = None
        else:
            raise ValueError(f"Unknown diffusion model: {model_type}")
                
    else:
        raise ValueError(f"Unknown model: {model_type}")

    return loss_fn


def build_optimizer(model, configs):
    model_type = configs["model"]["type"]
    optim_type = configs["train"]['optimizer']
    learning_rate = float(configs["train"]['learning_rate'])
    weight_decay = float(configs["train"].get("weight_decay", 0))
    adam_betas = (float(configs["train"].get("beta1", 0.9)), float(configs["train"].get("beta2", 0.999)))
    
    if model_type == 'vqvae':
        optimizer = {
            'optimizer_g': torch.optim.Adam(model['vqvae'].parameters(), lr=learning_rate, betas=adam_betas),
            'optimizer_d': torch.optim.Adam(model['discriminator'].parameters(), lr=learning_rate, betas=adam_betas)
        }
    else:
        if optim_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=adam_betas)
        elif optim_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optim_type}")

    return optimizer


def build_diffusion_scheduler(configs, device):
    from models.Diffusion.schedulers.gaussian_diffusion import linear_beta_schedule

    num_timesteps = int(configs["diffusion"]['num_timesteps'])

    betas = linear_beta_schedule(
        timesteps=num_timesteps,
        beta_start=float(configs["diffusion"]['beta_start']),
        beta_end=float(configs["diffusion"]['beta_end'])
    )

    scheduler = configs['diffusion'].get('diffuser', 'ddpm_scheduler')
    scheduler_kwargs = {"null_token_idx": configs.get("dataset", {}).get("num_classes", None)}
    
    if scheduler == 'ddpm_scheduler':
        from models.Diffusion.schedulers.ddpm_scheduler import DDPMScheduler
        scheduler_kwargs["variance_type"] = configs["diffusion"].get("variance_type", "fixed_small")
        DiffusionScheduler = DDPMScheduler
    elif scheduler == 'ddim_scheduler':
        from models.Diffusion.schedulers.ddim_scheduler import DDIMScheduler
        DiffusionScheduler = DDIMScheduler
    else:
        raise ValueError(f"Unknown diffusion scheduler: {scheduler}")

    diffusion = DiffusionScheduler(
        betas=betas,
        device=device,
        **scheduler_kwargs
    )
    
    return diffusion, num_timesteps