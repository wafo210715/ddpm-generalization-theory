import torch
from torchvision import transforms

from model.unet import DDPMUnet
from model.ddpm import DDPM

from tqdm.auto import tqdm
from util.img_utils import show_images, save_individual_images

from PIL import Image

def sample(num_samples: int, ddpm: DDPM, device: torch.device) -> list[Image.Image]:
    ddpm.eval()
    with torch.no_grad():
        x_t = torch.randn((num_samples, 1, 32, 32), device = device)
        
        for t in tqdm(range(ddpm.timesteps -1, -1, -1), desc="Sampling"):
            if t > 0:
                z = torch.randn_like(x_t, device=device)
            else:
                z = torch.zeros_like(x_t, device=device)
                
            t_tensor = torch.full((num_samples, ), t, device=device)
            x_t = ddpm.p_sample(x_t, t_tensor, z)
            
        x_t = x_t.clamp(0, 1)
        transform = transforms.ToPILImage()
        img_list = []
        for i in range(num_samples):
            img = transform(x_t[i].cpu())
            img_list.append(img)
        return img_list
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint path - can be changed via command line if needed
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "model_ckpts/mnist_32x32_steps50000_step10000.pth"

    # Load checkpoint and determine type
    checkpoint = torch.load(checkpoint_path, map_location=device)

    ddpm_unet = DDPMUnet(1, 64)
    ddpm = DDPM(1000, ddpm_unet)

    # Check if checkpoint is U-Net only (from dual GPU training) or full DDPM
    if "eps_model" in checkpoint or "beta" in checkpoint:
        # Full DDPM state dict
        ddpm.load_state_dict(checkpoint)
    else:
        # U-Net only (from dual GPU training) - load into eps_model
        ddpm.eps_model.load_state_dict(checkpoint)

    ddpm.to(device)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    img_list = sample(9, ddpm, device)
    show_images(img_list)
    save_individual_images(img_list)