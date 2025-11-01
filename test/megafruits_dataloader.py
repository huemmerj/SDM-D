"""
MegaFruits Dataset DataLoader
A PyTorch DataLoader for the MegaFruits dataset supporting strawberries, peaches, and blueberries.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json


class MegaFruitsDataset(Dataset):
    """
    PyTorch Dataset for MegaFruits dataset.
    
    Supports loading images with optional masks, labels, and annotations.
    
    Args:
        image_dir (str): Path to directory containing images
        mask_dir (str, optional): Path to directory containing segmentation masks
        label_dir (str, optional): Path to directory containing YOLO-format labels
        json_dir (str, optional): Path to directory containing JSON annotations
        transform (callable, optional): Optional transform to be applied on images
        mask_transform (callable, optional): Optional transform to be applied on masks
        image_ext (str): Image file extension (default: '.png')
        load_mode (str): Mode for loading data: 'image', 'image_mask', 'image_label', 'all' (default: 'image')
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        label_dir: Optional[str] = None,
        json_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        image_ext: str = '.png',
        load_mode: str = 'image'
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.label_dir = Path(label_dir) if label_dir else None
        self.json_dir = Path(json_dir) if json_dir else None
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_ext = image_ext
        self.load_mode = load_mode
        
        # Get list of image files
        self.image_files = sorted([
            f for f in self.image_dir.iterdir() 
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
            - 'image': PIL Image or transformed tensor
            - 'image_path': Path to the image file
            - 'image_name': Name of the image file (without extension)
            - 'mask': Segmentation mask (if available and load_mode requires it)
            - 'labels': YOLO format labels (if available and load_mode requires it)
            - 'annotations': JSON annotations (if available and load_mode requires it)
        """
        img_path = self.image_files[idx]
        img_name = img_path.stem
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        result = {
            'image_path': str(img_path),
            'image_name': img_name,
            'original_size': original_size
        }
        
        # Apply image transform
        if self.transform:
            image = self.transform(image)
        
        result['image'] = image
        
        # Load mask if requested
        if self.load_mode in ['image_mask', 'all'] and self.mask_dir:
            mask_path = self._find_mask(img_name)
            if mask_path and mask_path.exists():
                mask = self._load_mask(mask_path)
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                result['mask'] = mask
                result['mask_path'] = str(mask_path)
            else:
                result['mask'] = None
                result['mask_path'] = None
        
        # Load labels if requested
        if self.load_mode in ['image_label', 'all'] and self.label_dir:
            label_path = self.label_dir / f"{img_name}.txt"
            if label_path.exists():
                labels = self._load_yolo_labels(label_path)
                result['labels'] = labels
                result['label_path'] = str(label_path)
            else:
                result['labels'] = None
                result['label_path'] = None
        
        # Load JSON annotations if requested
        if self.load_mode == 'all' and self.json_dir:
            json_path = self.json_dir / f"{img_name}.json"
            if json_path.exists():
                annotations = self._load_json_annotations(json_path)
                result['annotations'] = annotations
                result['json_path'] = str(json_path)
            else:
                result['annotations'] = None
                result['json_path'] = None
        
        return result
    
    def _find_mask(self, img_name: str) -> Optional[Path]:
        """Find mask file for given image name."""
        # Check if mask_dir contains subdirectories with individual masks
        mask_subdir = self.mask_dir / img_name
        if mask_subdir.exists() and mask_subdir.is_dir():
            # Return the directory path (contains multiple mask files)
            return mask_subdir
        
        # Check for single mask file
        for ext in ['.png', '.npy', '.npz']:
            mask_path = self.mask_dir / f"{img_name}{ext}"
            if mask_path.exists():
                return mask_path
        
        return None
    
    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load segmentation mask from file or directory."""
        if mask_path.is_dir():
            # Load multiple masks and combine them
            mask_files = sorted(mask_path.glob('*.png'))
            if not mask_files:
                mask_files = sorted(mask_path.glob('*.npy'))
            
            if len(mask_files) == 0:
                return None
            
            # Load first mask to get shape
            first_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
            if first_mask is None:
                first_mask = np.load(str(mask_files[0]))
            
            combined_mask = np.zeros_like(first_mask, dtype=np.int32)
            
            # Combine all masks with unique IDs
            for i, mask_file in enumerate(mask_files, start=1):
                if mask_file.suffix == '.png':
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                else:
                    mask = np.load(str(mask_file))
                combined_mask[mask > 0] = i
            
            return combined_mask
        
        elif mask_path.suffix == '.png':
            return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        elif mask_path.suffix == '.npy':
            return np.load(str(mask_path))
        
        elif mask_path.suffix == '.npz':
            data = np.load(str(mask_path))
            # Return the first array in the npz file
            return data[data.files[0]]
        
        return None
    
    def _load_yolo_labels(self, label_path: Path) -> List[Dict[str, float]]:
        """Load YOLO format labels from txt file."""
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    labels.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
        return labels
    
    def _load_json_annotations(self, json_path: Path) -> Dict:
        """Load JSON annotations."""
        with open(json_path, 'r') as f:
            return json.load(f)


def create_megafruits_dataloader(
    image_dir: str,
    mask_dir: Optional[str] = None,
    label_dir: Optional[str] = None,
    json_dir: Optional[str] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    mask_transform: Optional[Callable] = None,
    load_mode: str = 'image',
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for MegaFruits dataset.
    
    Args:
        image_dir: Path to images directory
        mask_dir: Path to masks directory (optional)
        label_dir: Path to labels directory (optional)
        json_dir: Path to JSON annotations directory (optional)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        transform: Transform to apply to images
        mask_transform: Transform to apply to masks
        load_mode: What to load - 'image', 'image_mask', 'image_label', or 'all'
        **kwargs: Additional arguments to pass to DataLoader
    
    Returns:
        DataLoader instance
    """
    dataset = MegaFruitsDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        label_dir=label_dir,
        json_dir=json_dir,
        transform=transform,
        mask_transform=mask_transform,
        load_mode=load_mode
    )
    
    # Custom collate function to handle variable-sized data
    def collate_fn(batch):
        # Don't stack images if they have different sizes
        # Return as list of dictionaries instead
        return batch
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )


# Example usage and utility functions
if __name__ == "__main__":
    # Example 1: Load only images
    print("Example 1: Loading images only")
    dataloader = create_megafruits_dataloader(
        image_dir='./Images/strawberry',
        batch_size=2,
        shuffle=False,
        load_mode='image'
    )
    
    for batch in dataloader:
        for item in batch:
            print(f"Image: {item['image_name']}, Size: {item['original_size']}")
        break
    
    # Example 2: Load images with masks
    print("\nExample 2: Loading images with masks")
    dataloader = create_megafruits_dataloader(
        image_dir='./Images/strawberry',
        mask_dir='./output/strawberry/mask',
        batch_size=1,
        load_mode='image_mask'
    )
    
    for batch in dataloader:
        for item in batch:
            print(f"Image: {item['image_name']}")
            if item['mask'] is not None:
                print(f"  Mask shape: {item['mask'].shape}")
            else:
                print("  No mask found")
        break
    
    # Example 3: Load images with labels
    print("\nExample 3: Loading images with labels")
    dataloader = create_megafruits_dataloader(
        image_dir='./Images/strawberry',
        label_dir='./output/strawberry/labels',
        batch_size=1,
        load_mode='image_label'
    )
    
    for batch in dataloader:
        for item in batch:
            print(f"Image: {item['image_name']}")
            if item['labels'] is not None:
                print(f"  Number of labels: {len(item['labels'])}")
                for label in item['labels'][:3]:  # Print first 3 labels
                    print(f"    Class {label['class_id']}: bbox center=({label['x_center']:.3f}, {label['y_center']:.3f})")
            else:
                print("  No labels found")
        break
    
    # Example 4: Load everything
    print("\nExample 4: Loading images with all annotations")
    dataloader = create_megafruits_dataloader(
        image_dir='./Images/strawberry',
        mask_dir='./output/strawberry/mask',
        label_dir='./output/strawberry/labels',
        json_dir='./output/strawberry/json',
        batch_size=1,
        load_mode='all'
    )
    
    for batch in dataloader:
        for item in batch:
            print(f"Image: {item['image_name']}")
            print(f"  Has mask: {item['mask'] is not None}")
            print(f"  Has labels: {item['labels'] is not None}")
            print(f"  Has annotations: {item['annotations'] is not None}")
        break
