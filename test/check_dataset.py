"""
Simple script to test and check the MegaFruits dataset using the dataloader.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from megafruits_dataloader import create_megafruits_dataloader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_batch(batch, max_images=4):
    """Visualize a batch of images with their annotations."""
    n_images = min(len(batch), max_images)
    
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    if n_images == 1:
        axes = [axes]
    
    for idx, (ax, item) in enumerate(zip(axes, batch)):
        # Display image
        if isinstance(item['image'], np.ndarray):
            ax.imshow(item['image'])
        else:
            ax.imshow(item['image'])
        
        ax.set_title(f"{item['image_name']}\n{item['original_size']}")
        ax.axis('off')
        
        # Add label count if available
        if 'labels' in item and item['labels'] is not None:
            ax.text(0.05, 0.95, f"Labels: {len(item['labels'])}", 
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='top')
    
    plt.tight_layout()
    return fig


def check_dataset_statistics(dataloader):
    """Print statistics about the dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total_images = len(dataloader.dataset)
    print(f"Total images: {total_images}")
    
    # Count items with masks, labels, annotations
    has_mask = 0
    has_labels = 0
    has_annotations = 0
    total_objects = 0
    
    image_sizes = []
    
    for batch in dataloader:
        for item in batch:
            image_sizes.append(item['original_size'])
            
            if 'mask' in item and item['mask'] is not None:
                has_mask += 1
            
            if 'labels' in item and item['labels'] is not None:
                has_labels += 1
                total_objects += len(item['labels'])
            
            if 'annotations' in item and item['annotations'] is not None:
                has_annotations += 1
    
    print(f"\nImages with masks: {has_mask}/{total_images}")
    print(f"Images with labels: {has_labels}/{total_images}")
    print(f"Images with JSON annotations: {has_annotations}/{total_images}")
    
    if has_labels > 0:
        print(f"Average objects per labeled image: {total_objects/has_labels:.2f}")
    
    # Image size statistics
    if image_sizes:
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        print(f"\nImage size statistics:")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.1f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.1f}")


def main():
    """Main function to test the dataloader."""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    image_dir = base_dir / "Images" / "strawberry"
    mask_dir = base_dir / "output" / "strawberry" / "mask"
    label_dir = base_dir / "output" / "strawberry" / "labels"
    json_dir = base_dir / "output" / "strawberry" / "json"
    
    print("MegaFruits Dataset Checker")
    print("="*60)
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Label directory: {label_dir}")
    print(f"JSON directory: {json_dir}")
    
    # Check which directories exist
    print("\nDirectory status:")
    print(f"  Images exist: {image_dir.exists()}")
    print(f"  Masks exist: {mask_dir.exists()}")
    print(f"  Labels exist: {label_dir.exists()}")
    print(f"  JSON exist: {json_dir.exists()}")
    
    if not image_dir.exists():
        print("\nError: Image directory not found!")
        return
    
    # Create dataloader based on what's available
    load_mode = 'image'
    if mask_dir.exists() and label_dir.exists():
        load_mode = 'all'
    elif mask_dir.exists():
        load_mode = 'image_mask'
    elif label_dir.exists():
        load_mode = 'image_label'
    
    print(f"\nLoad mode: {load_mode}")
    
    # Create dataloader
    dataloader = create_megafruits_dataloader(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir) if mask_dir.exists() else None,
        label_dir=str(label_dir) if label_dir.exists() else None,
        json_dir=str(json_dir) if json_dir.exists() else None,
        batch_size=2,
        shuffle=False,
        load_mode=load_mode
    )
    
    # Print dataset statistics
    check_dataset_statistics(dataloader)
    
    # Show first batch
    print("\n" + "="*60)
    print("FIRST BATCH SAMPLE")
    print("="*60)
    
    for batch in dataloader:
        for i, item in enumerate(batch):
            print(f"\nImage {i+1}: {item['image_name']}")
            print(f"  Path: {item['image_path']}")
            print(f"  Size: {item['original_size']}")
            
            if 'mask' in item and item['mask'] is not None:
                print(f"  Mask shape: {item['mask'].shape}")
                print(f"  Unique mask values: {len(np.unique(item['mask']))}")
            
            if 'labels' in item and item['labels'] is not None:
                print(f"  Number of labels: {len(item['labels'])}")
                for j, label in enumerate(item['labels'][:5]):  # Show first 5
                    print(f"    Label {j+1}: class={label['class_id']}, "
                          f"center=({label['x_center']:.3f}, {label['y_center']:.3f}), "
                          f"size=({label['width']:.3f}, {label['height']:.3f})")
                if len(item['labels']) > 5:
                    print(f"    ... and {len(item['labels'])-5} more")
            
            if 'annotations' in item and item['annotations'] is not None:
                print(f"  Annotations keys: {list(item['annotations'].keys())}")
        
        break  # Only show first batch
    
    print("\n" + "="*60)
    print("Dataset check complete!")
    print("="*60)


if __name__ == "__main__":
    main()
