"""Script for Self-Supervised Descriptor for Image Copy Detection (SSCD)."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
from torch_utils import distributed as dist
from training import dataset
from torchvision import transforms
import wget
from PIL import Image

#----------------------------------------------------------------------------

def TransformSamples(samples, transform):
    sscd_samples = []
    for sample in samples:
      sscd_samples.append(transform(sample)[None, :])
    return torch.cat(sscd_samples, dim=0)


@click.group()
def main():
    """Calculate Self-Supervised Descriptor for Image Copy Detection (SSCD).
    The original github is https://github.com/facebookresearch/sscd-copy-detection

    Examples:

    \b
    # Calculate SSCD feature 
    torchrun --standalone --nproc_per_node=1 edm/sscd.py feature --images ./evaluation/ddpm-dim128-n16384 --features ./evaluation/sscd-dim128-n16384.npz

    torchrun --standalone --nproc_per_node=1 edm/sscd.py feature --images ./evaluation/ddpm-dim64-n16384 --features ./evaluation/sscd-dim64-n16384.npz

    torchrun --standalone --nproc_per_node=1 edm/sscd.py feature --images ./evaluation/generalization/ --features ./evaluation/sscd-generalization.npz

    torchrun --standalone --nproc_per_node=1 edm/sscd.py feature --images datasets/synthetic-cifar10-32x32-n16384.zip --features ./evaluation/sscd-training-dataset-synthetic-cifar10-32x32-n16384.npz

    \b
    # Compute reproducibility score
    python edm/sscd.py rpscore --source ./evaluation/sscd-dim128-n16384.npz --target ./evaluation/sscd-dim64-n16384.npz

    # Compute generalization score
    python edm/sscd.py rpscore --source ./evaluation/sscd-dim128-n16384.npz --target ./evaluation/sscd-generalization.npz

    # Compute memorization score
    python edm/sscd.py mscore --source ./evaluation/sscd-dim128-n16384.npz --target ./evaluation/sscd-training-dataset-synthetic-cifar10-32x32-n16384.npz
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|NPZ',              type=str, required=True)
@click.option('--features', 'features_path', help='Path to save features', metavar='NPZ',              type=str, required=True)
def feature(image_path, features_path):
    """Calculate SSCD features for a given set of images."""
    if not os.path.exists("./pretrainedmodels"):
        os.makedirs("./pretrainedmodels")
    if not os.path.exists("./pretrainedmodels/sscd_disc_large.torchscript.pt"):
        wget.download("https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt", "./pretrainedmodels/sscd_disc_large.torchscript.pt")
    sscd_model = torch.jit.load("./pretrainedmodels/sscd_disc_large.torchscript.pt")
    sscd_model = sscd_model.to(device=f"cuda:0")
    sscd_model.eval()
    sscd_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        # transform to float tensor
        # transforms.Lambda(lambda x: x.float()),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)
    ])
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=image_path)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataloader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=64, **data_loader_kwargs)
    sscd_features = []
    for x_batch, _ in tqdm.tqdm(dataloader):
        x_sscd = TransformSamples(x_batch, sscd_transform).to(device=f"cuda:0")
        sscd_feature = sscd_model(x_sscd).detach().cpu()
        sscd_features.append(sscd_feature)
    sscd_features = torch.cat(sscd_features, dim=0)
    np.savez(features_path, features=sscd_features.numpy())
# #----------------------------------------------------------------------------

@main.command()
@click.option('--source', 'source_path', help='Path to source sscd feature', metavar='NPZ', type=str, required=True)
@click.option('--target', 'target_path', help='Path to source sscd feature', metavar='NPZ', type=str, required=True)
@click.option('--t', 'threshold', help='threshold for sscd similarity', type=float, default=0.6)

def rpscore(source_path, target_path, threshold):
    """Calculate reproducibility score between source images and targe images."""
    source_features = np.load(source_path)["features"]
    target_features = np.load(target_path)["features"]
    similarity = (source_features * target_features).sum(axis=1)
    rpscore = (similarity > threshold).mean()
    print('RP score = ', rpscore)
    return rpscore

@main.command()
@click.option('--source', 'source_path', help='Path to source sscd feature', metavar='NPZ', type=str, required=True)
@click.option('--target', 'target_path', help='Path to target sscd feature', metavar='NPZ', type=str, required=True)
@click.option('--source_img', 'source_img_path', help='Path to source image', type=str, default=None)
@click.option('--target_img', 'target_img_path', help='Path to target image', type=str, default=None)
@click.option('--vis_mem_sample', 'vis_mem_sample', help='Visualize memorization sample', type=bool, default=False)
@click.option('--num_mem_sample', 'num_mem_sample', help='Number of memorization sample', type=int, default=16)
@click.option('--mem_sample_path', 'mem_sample_path', help='Path to memorization sample', type=str, default=None)
@click.option('--t', 'threshold', help='threshold for sscd similarity', type=float, default=0.6)
@click.option('--k', 'k', help='number of similar images', type=int, default=3) 

def mscore(source_path, target_path, source_img_path, target_img_path, vis_mem_sample, num_mem_sample, mem_sample_path, threshold, k):
    """Calculate reproducibility score between source images and targe images."""
    bs = 128
    source_features = np.load(source_path)["features"][:, None, :]  
    target_features = np.load(target_path)["features"][None, :, :]
    rpscore = 0
    total_sample = source_features.shape[0]
    for idx in tqdm.tqdm(range(total_sample//bs + 1)):
        similarity = (source_features[idx*bs: (idx + 1)*bs, :] * target_features).sum(axis=2).max(axis=1)
        rpscore += (similarity > threshold).sum()
    rpscore = rpscore/total_sample
    print('M score = ', 1 - rpscore)

    if vis_mem_sample:
        similarity = (source_features[:num_mem_sample, :] * target_features).sum(axis=2)
        
        # Get top k similar images for each generated sample
        top_indices = np.argsort(similarity, axis=1)[:, -k:][:, ::-1]  # Shape: (num_mem_sample, k)
        # Load images and create grid
        img_size = 32  # Adjust based on your image size
        grid_width = k + 1  # 1 generated image + k similar training images
        grid_height = num_mem_sample
        
        # Create a large canvas
        grid_image = Image.new('RGB', (grid_width * img_size, grid_height * img_size))
        
        for row_idx in range(num_mem_sample):
            # Load generated image (source)
            gen_img_path = os.path.join(source_img_path, f"{row_idx:06d}.png")
            if os.path.exists(gen_img_path):
                gen_img = Image.open(gen_img_path).resize((img_size, img_size))
                grid_image.paste(gen_img, (0, row_idx * img_size))
            
            # Load top  similar training images (target)
            for col_idx, target_idx in enumerate(top_indices[row_idx]):
                target_img_path_full = os.path.join(target_img_path, f"{target_idx:06d}.png")
                if os.path.exists(target_img_path_full):
                    target_img = Image.open(target_img_path_full).resize((img_size, img_size))
                    grid_image.paste(target_img, ((col_idx + 1) * img_size, row_idx * img_size))
        
        # Save the grid image
        grid_image.save(mem_sample_path)
        print(f"Saved visualization grid to {mem_sample_path}")
        
    return rpscore


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
