"""
Utility functions for displaying and saving images.
"""
import os
from typing import List
from PIL import Image
import matplotlib.pyplot as plt


def show_images(images: List[Image.Image], nrow: int = 3, title: str = "Generated Images") -> None:
    """
    Display a list of PIL images in a grid.

    Args:
        images: List of PIL Image objects to display
        nrow: Number of images per row
        title: Title for the figure
    """
    n_images = len(images)
    ncol = (n_images + nrow - 1) // nrow

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    fig.suptitle(title, fontsize=14)

    if ncol == 1 and nrow == 1:
        axes.imshow(images[0], cmap='gray', vmin=0, vmax=255)
        axes.axis('off')
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, ax in enumerate(axes):
            if i < n_images:
                ax.imshow(images[i], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def save_images(images: List[Image.Image], output_dir: str = "outputs", filename: str = "generated_images.png") -> None:
    """
    Save a list of PIL images as a single grid image.

    Args:
        images: List of PIL Image objects to save
        output_dir: Directory to save the output image
        filename: Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)

    n_images = len(images)
    nrow = 3
    ncol = (n_images + nrow - 1) // nrow

    img_width, img_height = images[0].size
    grid_width = nrow * img_width
    grid_height = ncol * img_height

    # Create a new blank image for the grid
    grid_image = Image.new('L', (grid_width, grid_height))

    # Paste each image into the grid
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))

    output_path = os.path.join(output_dir, filename)
    grid_image.save(output_path)
    print(f"Images saved to: {output_path}")


def save_individual_images(images: List[Image.Image], output_dir: str = "outputs", prefix: str = "image") -> List[str]:
    """
    Save each image individually.

    Args:
        images: List of PIL Image objects to save
        output_dir: Directory to save the output images
        prefix: Prefix for each filename

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    for idx, img in enumerate(images):
        filename = f"{prefix}_{idx:04d}.png"
        output_path = os.path.join(output_dir, filename)
        img.save(output_path)
        saved_paths.append(output_path)

    print(f"Saved {len(saved_paths)} individual images to: {output_dir}")
    return saved_paths
