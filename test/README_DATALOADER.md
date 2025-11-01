# MegaFruits DataLoader

A flexible PyTorch DataLoader for the MegaFruits dataset supporting images, masks, labels, and annotations.

## Features

✅ Load images from directory  
✅ Support for segmentation masks (single files or directories of masks)  
✅ Support for YOLO-format labels  
✅ Support for JSON annotations  
✅ Flexible loading modes (image only, with masks, with labels, or all)  
✅ Custom transforms for images and masks  
✅ Batch processing support  
✅ Compatible with PyTorch training pipelines

## Installation

No additional packages required beyond the project requirements:

```bash
pip install torch torchvision pillow opencv-python numpy
```

## Quick Start

### 1. Basic Usage - Load Images Only

```python
from megafruits_dataloader import create_megafruits_dataloader

# Create a simple dataloader for images
dataloader = create_megafruits_dataloader(
    image_dir='./Images/strawberry',
    batch_size=4,
    shuffle=True,
    load_mode='image'
)

# Iterate through the data
for batch in dataloader:
    for item in batch:
        image = item['image']  # PIL Image
        name = item['image_name']
        size = item['original_size']
        print(f"Processing {name}: {size}")
```

### 2. Load Images with Masks

```python
dataloader = create_megafruits_dataloader(
    image_dir='./Images/strawberry',
    mask_dir='./output/strawberry/mask',
    batch_size=2,
    load_mode='image_mask'
)

for batch in dataloader:
    for item in batch:
        image = item['image']
        mask = item['mask']  # numpy array or None
        if mask is not None:
            print(f"Mask shape: {mask.shape}")
```

### 3. Load Images with YOLO Labels

```python
dataloader = create_megafruits_dataloader(
    image_dir='./Images/strawberry',
    label_dir='./output/strawberry/labels',
    batch_size=2,
    load_mode='image_label'
)

for batch in dataloader:
    for item in batch:
        labels = item['labels']  # List of dicts or None
        if labels:
            for label in labels:
                class_id = label['class_id']
                x_center = label['x_center']
                y_center = label['y_center']
                width = label['width']
                height = label['height']
```

### 4. Load Everything

```python
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
        image = item['image']
        mask = item['mask']
        labels = item['labels']
        annotations = item['annotations']
```

## Using with Transforms

### Image Transforms (torchvision)

```python
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataloader = create_megafruits_dataloader(
    image_dir='./Images/strawberry',
    transform=transform,
    batch_size=8,
    load_mode='image'
)

# Now item['image'] will be a normalized tensor
```

### Custom Mask Transform

```python
def mask_transform(mask):
    """Convert mask to tensor and resize."""
    import torch
    from torchvision.transforms.functional import resize
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add channel dim
    mask_tensor = resize(mask_tensor, [512, 512])
    return mask_tensor

dataloader = create_megafruits_dataloader(
    image_dir='./Images/strawberry',
    mask_dir='./output/strawberry/mask',
    mask_transform=mask_transform,
    load_mode='image_mask'
)
```

## Data Structure

Each item in the batch is a dictionary containing:

```python
{
    'image': PIL.Image or Tensor,           # The image (depends on transform)
    'image_path': str,                       # Full path to image file
    'image_name': str,                       # Image filename without extension
    'original_size': tuple,                  # (width, height) of original image
    'mask': np.ndarray or None,             # Segmentation mask (if load_mode requires)
    'mask_path': str or None,               # Path to mask file
    'labels': List[dict] or None,           # YOLO labels (if load_mode requires)
    'label_path': str or None,              # Path to label file
    'annotations': dict or None,            # JSON annotations (if load_mode requires)
    'json_path': str or None                # Path to JSON file
}
```

### YOLO Label Format

Each label in the `labels` list is a dictionary:

```python
{
    'class_id': int,      # Class ID
    'x_center': float,    # Normalized x-center (0-1)
    'y_center': float,    # Normalized y-center (0-1)
    'width': float,       # Normalized width (0-1)
    'height': float       # Normalized height (0-1)
}
```

## Load Modes

- **`'image'`**: Load only images
- **`'image_mask'`**: Load images and masks
- **`'image_label'`**: Load images and YOLO labels
- **`'all'`**: Load images, masks, labels, and JSON annotations

## Advanced Usage

### Multi-Worker Data Loading

```python
dataloader = create_megafruits_dataloader(
    image_dir='./Images/strawberry',
    batch_size=16,
    num_workers=4,  # Use 4 worker processes
    shuffle=True,
    load_mode='image'
)
```

### Training Loop Example

```python
import torch
import torch.nn as nn
from torchvision import transforms, models

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Create dataloader with transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataloader = create_megafruits_dataloader(
    image_dir='./Images/strawberry',
    transform=transform,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        # Since batch is a list of dicts, extract images
        images = torch.stack([item['image'] for item in batch]).to(device)

        # Forward pass
        outputs = model(images)

        # Your training code here...
```

### Integration with SDM.py

You can integrate this dataloader into the existing SDM pipeline:

```python
from megafruits_dataloader import MegaFruitsDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = MegaFruitsDataset(
    image_dir='./Images/strawberry',
    load_mode='image'
)

# Process with SDM
for i in range(len(dataset)):
    item = dataset[i]
    image_path = item['image_path']
    # Process with your SDM pipeline...
```

## Testing

Run the test script to check your dataset:

```bash
cd test
python check_dataset.py
```

This will:

- Check which directories exist
- Count images, masks, labels, and annotations
- Display statistics about the dataset
- Show sample data from the first batch

## Supported File Formats

- **Images**: `.png`, `.jpg`, `.jpeg`, `.bmp`
- **Masks**: `.png`, `.npy`, `.npz` (single file or directory of mask files)
- **Labels**: `.txt` (YOLO format)
- **Annotations**: `.json`

## Notes

- The dataloader automatically handles missing masks, labels, or annotations
- Masks can be stored as single files or as directories containing multiple mask files
- When masks are in a directory, they are combined with unique IDs
- Batch size is flexible but note that images may have different sizes
- Use transforms to resize images if you need uniform sizes for batching

## Examples

See:

- `megafruits_dataloader.py` - Main dataloader implementation with usage examples
- `check_dataset.py` - Testing and validation script

## Troubleshooting

**Q: No images found?**  
A: Check that `image_dir` path is correct and contains image files

**Q: Masks not loading?**  
A: Ensure `mask_dir` path is correct and set `load_mode='image_mask'` or `load_mode='all'`

**Q: Memory issues?**  
A: Reduce `batch_size` or use fewer `num_workers`

**Q: Different image sizes?**  
A: Use transforms to resize images to a uniform size for batching

## License

Same as parent project (SDM-D)
