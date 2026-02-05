from xml.parsers.expat import model
import torch
import os
import argparse
import yaml
from models.build import build_model, build_loss_function, build_optimizer
from datasets.build import build_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.util_makegif import TrainingProcessRecorder
from utils.util_visualization import visualize_generated_samples, visualize_reconstructions, visualize_loss_curve
from utils.util_save import save_vae_model_checkpoint, save_gan_model_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args


def yaml_loader(configs_path):
    with open(configs_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs
    

def main():
    """
        Loading configurations
    """
    args = parse_args()
    configs = yaml_loader(args.config)
    
    task = configs["task"]
    
    os.makedirs(os.path.join("output", configs["task_name"], "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join("output", configs["task_name"], "visualization", "train"), exist_ok=True)

    if task == 'vae':
        os.makedirs(os.path.join("output", configs["task_name"], "visualization", "test"), exist_ok=True)

    """
        Building dataset and dataloaders
    """
    batch_size = configs["train"]["batch_size"]
    num_workers = configs["train"]["num_workers"]
    
    train_dataset, test_dataset = build_dataset(configs)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    """
        Building model
    """
    in_channels = int(configs["model"]['in_channels'])
    img_size = int(configs["model"]['img_size'])

    model = build_model(configs)
    
    """
        Training model
    """
    num_epochs = int(configs["train"]["epochs"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    if task == 'vae':
        optimizer = build_optimizer(model, configs)
    elif task == 'gan':
        optimizer_G = build_optimizer(model.netG, configs)
        optimizer_D = build_optimizer(model.netD, configs)
        
    criterion = build_loss_function(configs)
    
    recorder = TrainingProcessRecorder(configs, device)
        
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        pbar = tqdm(train_dataloader, desc=f"Epoch [{epoch}/{num_epochs}]")
        
        train_loss_sum = 0.0
        train_loss_history = []
        
        if task == 'vae':
            for inputs, _ in pbar:
                inputs = inputs.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item()
                
                batch_avg_loss = loss.item() / len(inputs)
                pbar.set_postfix({'loss': f"{batch_avg_loss:.4f}"})
            
            avg_train_loss = train_loss_sum / len(train_dataloader.dataset)
            train_loss_history.append(avg_train_loss)
        
            # -------------------------------------------------------
            # Test Loop (Validation)
            # -------------------------------------------------------
            model.eval()
            test_loss_sum = 0.0
            test_loss_history = []
            
            recorder.record_frame(model.decoder)
            
            with torch.no_grad():
                for test_inputs, _ in test_dataloader:
                    test_inputs = test_inputs.to(device)
                    test_outputs = model(test_inputs)
                    test_loss = criterion(test_outputs, test_inputs)
                    test_loss_sum += test_loss.item()
                    
            avg_test_loss = test_loss_sum / len(test_dataloader.dataset)
            test_loss_history.append(avg_test_loss)
            
            print(f"Epoch [{epoch}/{num_epochs}] Done. | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
            

            if epoch % 10 == 0:            
                visualize_reconstructions(model, configs, train_dataloader, device, epoch, train=True)
                visualize_reconstructions(model, configs, test_dataloader, device, epoch, train=False)
                
                save_vae_model_checkpoint(model, optimizer, avg_train_loss, configs, epoch=epoch)
 
        elif task == 'gan':
            loss_D_sum = 0.0
            loss_G_sum = 0.0
            loss_G_history = []
            loss_D_history = []
            netG = model.netG
            netD = model.netD
            
            for real_images, _ in pbar:
                real_images = real_images.to(device)
                batch_size = real_images.size(0)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                z = torch.randn(batch_size, netG.latent_dim).to(device)
                fake_images = netG(z)
                label_real = torch.ones(batch_size).to(device)
                label_fake = torch.zeros(batch_size).to(device)
                
                output_real = netD(real_images)
                output_fake = netD(fake_images.detach()) # Generator no update
                
                loss_D_real = criterion(output_real, label_real)
                loss_D_fake = criterion(output_fake, label_fake)
                
                loss_D = (loss_D_real + loss_D_fake) / 2
                loss_D.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                
                output = netD(fake_images)
                
                loss_G = criterion(output, label_real) # Trick: want fake to be real
                loss_G.backward()
                optimizer_G.step()

                loss_D_sum += loss_D.item()
                loss_G_sum += loss_G.item()
                loss_D_history.append(loss_D.item())
                loss_G_history.append(loss_G.item())
                
                pbar.set_postfix({'loss_D': f"{loss_D.item():.4f}", 'loss_G': f"{loss_G.item():.4f}"})
        
            avg_loss_D = loss_D_sum / len(train_dataloader)
            avg_loss_G = loss_G_sum / len(train_dataloader)
            
            print(f"Epoch [{epoch}/{num_epochs}] Done. | Loss D: {avg_loss_D:.4f} | Loss G: {avg_loss_G:.4f}")
            recorder.record_frame(model.netG)
            
            if epoch % 10 == 0:  
                visualize_generated_samples(netG, configs, device, epoch=epoch)           
                save_gan_model_checkpoint(model, optimizer_G, optimizer_D, avg_loss_G, avg_loss_D, configs, epoch=epoch)
 
    recorder.save_gif()
    
    model.eval()
    if task == 'vae':
        visualize_generated_samples(model.decoder, configs, device)
        visualize_reconstructions(model, configs, test_dataloader, device)
        visualize_loss_curve(configs, train_loss_history, test_loss_history)
        save_vae_model_checkpoint(model, optimizer, avg_train_loss, configs, num_epochs, final=True)
    elif task == 'gan':
        visualize_generated_samples(model.netG, configs, device)
        save_gan_model_checkpoint(model, optimizer_G, optimizer_D, avg_loss_G, avg_loss_D, configs, num_epochs, final=True)
    
if __name__ == "__main__":
    main()
    