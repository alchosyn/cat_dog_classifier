#Project Initialization Import Area
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from matplotlib.style.core import available
from sympy import Number
#pip install torchvision
from torchvision import datasets, transforms, models
import imageio
import time
import warnings
warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image

#Data Reading
data_dir = './datasets_catdog/'
train_dir = data_dir + 'train/'
valid_dir = data_dir + 'valid/'

#Image Data Preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256), #256*256
        transforms.RandomRotation(30),  #(-30,30)
        transforms.CenterCrop(224),  #Center crop, processed image 224×224
        transforms.RandomHorizontalFlip(p=0.5),  #50% probability of random vertical flip
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  #Color adjustment, adapts to various ambient lighting conditions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #Mean and standard deviation
    ]),
    'valid': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

#Folder specification
batch_size = 64
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x] ) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

print(f"train number: {dataset_sizes['train']}")
print(f"valid number: {dataset_sizes['valid']}")
print(f"type number: {class_names}")

# === quick sanity check ===
xb, yb = next(iter(dataloaders['train']))
print("one batch:", xb.shape, yb.shape, xb.dtype, xb.min().item(), xb.max().item())

#Load the model provided in the model file, using the pre-trained weights as initialization parameters.
model_name = 'resnet'
feature_extract = True

#Should GPU training be used
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available. Training on CPU.")
else:
    print("CUDA is available. Training on GPU.")

device = torch.device("cuda: 0" if train_on_gpu else "cpu")

model_ft = models.resnet18()
model_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

param_to_update = []
for name, param in model_ft.named_parameters():
    if param.requires_grad == True:
        param_to_update.append(param)
    print("\t", name, param.requires_grad)


from torchvision.models import ResNet18_Weights

# Should we only perform feature extraction (freeze the main branch)
feature_extract = True
set_parameter_requires_grad(model_ft, feature_extract)
# Replace Category Header (Based on Number of Categories in Dataset)
num_ftrs = model_ft.fc.in_features
num_classes = len(class_names)   # e.g. 2（cat/dog）
model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
# Move to device
model_ft = model_ft.to(device)
# Print the complete structure
print(model_ft)
print("Class names:", class_names)
print("class_to_idx:", image_datasets['train'].class_to_idx)

import torchvision.utils as vutils

# Retrieve a batch from DataLoader
inputs, classes = next(iter(dataloaders['train']))

print("Batch Shape:", inputs.shape)  # torch.Size([64, 3, 224, 224])
print("Category Index:", classes[:8])

# Inverse Normalization Function
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean   # Anti-standardization
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Display a batch of images
out = vutils.make_grid(inputs[:8], nrow=4)
imshow(out, title=[class_names[x] for x in classes[:8]])
plt.show()

#Single Image Preprocessing Effect
from PIL import Image

img_path = "./datasets_catdog/train/cat/cat.0.jpg"
img = Image.open(img_path).convert("RGB")

transformed_img = data_transforms['train'](img)

print("Transformed Tensor Shape:", transformed_img.shape)

  # Anti-Standardization Re-Display
img_np = transformed_img.numpy().transpose((1, 2, 0))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_np = std * img_np + mean
plt.imshow(np.clip(img_np, 0, 1))
plt.show()

import os
import torch
from torchvision.utils import save_image

# === 1. Specify output folder ===
save_dir = r".\processed_samples"
os.makedirs(save_dir, exist_ok=True)

# === 2. Define denormalization for restoring image colors ===
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# === 3. Initialize the counter ===
img_counter = 0
skipped_counter = 0

# === 4. Iterate through the entire training set DataLoader ===
print("\n🚀 Begin preprocessing the entire training dataset...")
for batch_i, (inputs, classes) in enumerate(dataloaders['train']):
    for i in range(len(inputs)):
        # Anti-standardization
        img = inputs[i] * std + mean
        img = torch.clamp(img, 0, 1)

        # Retrieve category name
        label_name = class_names[classes[i]]

        # Category Folder
        class_folder = os.path.join(save_dir, label_name)
        os.makedirs(class_folder, exist_ok=True)

        # Generate and save file name (named sequentially)
        file_name = f"{label_name}_{img_counter:05d}.png"
        save_path = os.path.join(class_folder, file_name)

        # If the file already exists, skip it.
        if os.path.exists(save_path):
            skipped_counter += 1
            continue

        # Save Image
        save_image(img, save_path)
        img_counter += 1

    print(f"batch {batch_i+1}/{len(dataloaders['train'])} Processed。")

print(f"\n✅ Training set preprocessing completed. Total saved: {img_counter} new images，Skip {skipped_counter} existing files。")

# === Iterate through the validation set  ===
print("\n🚀 Begin preprocessing the validation set data...")
for batch_i, (inputs, classes) in enumerate(dataloaders['valid']):
    for i in range(len(inputs)):
        img = inputs[i] * std + mean
        img = torch.clamp(img, 0, 1)
        label_name = class_names[classes[i]]
        class_folder = os.path.join(save_dir, f"valid_{label_name}")
        os.makedirs(class_folder, exist_ok=True)
        file_name = f"{label_name}_{batch_i:04d}_{i:03d}.png"
        save_path = os.path.join(class_folder, file_name)

        if os.path.exists(save_path):
            skipped_counter += 1
            continue

        save_image(img, save_path)
        img_counter += 1

    print(f"Verification Batch {batch_i+1}/{len(dataloaders['valid'])} Processed。")

print(f"\n🎯 All preprocessing completed: {img_counter} images generated in total，Skip {skipped_counter} existing files")
print(f"📂 The file has been saved to：{save_dir}")

import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch
from tqdm import tqdm  # pip install tqdm

# === 0) Output root directory ===
OUT_ROOT = r".\processed_samples"

# === 1) Deterministic preprocessing (no random augmentations) ===
# Use this transform only for exporting; keep your original random transforms for training.
export_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize to 224x224
    transforms.ToTensor(),
    # If you want normalized tensors, uncomment the following line:
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def export_split(split_name: str):
    """
    Export one dataset split ('train' or 'valid') from the ImageFolder dataset.
    Each image is processed once and saved under OUT_ROOT/split/class/filename.png.
    """
    ds = image_datasets[split_name]  # torchvision.datasets.ImageFolder
    classes = ds.classes
    samples = ds.samples  # [(path, class_idx), ...]

    out_split_dir = os.path.join(OUT_ROOT, split_name)
    safe_mkdir(out_split_dir)

    # Create subfolders for each class
    for cname in classes:
        safe_mkdir(os.path.join(out_split_dir, cname))

    saved, skipped, errors = 0, 0, 0

    for src_path, cls_idx in tqdm(samples, desc=f"Exporting {split_name}", ncols=100):
        cname = classes[cls_idx]
        base = os.path.splitext(os.path.basename(src_path))[0]
        dst_path = os.path.join(out_split_dir, cname, f"{base}.png")

        if os.path.exists(dst_path):
            skipped += 1
            continue

        try:
            img = Image.open(src_path).convert("RGB")
            tensor = export_transform(img)
            save_image(tensor, dst_path)
            saved += 1
        except Exception as e:
            errors += 1
            print(f"[WARN] Failed on {src_path}: {e}")

    print(f"[{split_name}] saved={saved}, skipped={skipped}, errors={errors}, out_dir={out_split_dir}")

# === 2) Run full export for train and valid sets ===
export_split('train')
export_split('valid')
print(f"✅ Export complete. All processed images saved to: {OUT_ROOT}")
