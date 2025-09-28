# DATASET -> https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset


# Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms.functional as TF
import cv2


# Use GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Loading dataset
project_root = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(project_root, '..', 'archive', 'supervisely_person_clean_2667_img', 'supervisely_person_clean_2667_img', 'images')
mask_dir = os.path.join(project_root, '..', 'archive', 'supervisely_person_clean_2667_img', 'supervisely_person_clean_2667_img', 'masks')

image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))


# Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])  # same name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        # Always resize both image and mask to (256, 256)
        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)

        if self.transform:
            image = self.transform(image)
        mask = TF.to_tensor(mask).float()  # convert to tensor
        return image, mask


# Transforms and DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)


# Loading Pretrained DeepLabV3 Model
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # binary mask
model = model.to(device)


# Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Training Loop
if not (os.path.exists("segment-training.pth")):
    print("Training model...")
    # --- Training loop ---
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}", flush=True)
        model.train()
        total_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            print(f"  Batch {i+1}/{len(train_loader)}", flush=True)
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}", flush=True)

    torch.save(model.state_dict(), "segment-training.pth")
    print("Training complete and weights saved.")
else:
    print("Weights found, skipping training.")


# Loading weights before inference
model.load_state_dict(torch.load("segment-training.pth"))
model.eval()


# Visualize Predictions
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_and_visualize_cv2(img_path):


    model.load_state_dict(torch.load("segment-training.pth"))
    model.eval()


    # 1. Load image using OpenCV
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # 2. Apply transforms
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # 3. Model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    # 4. Resize prediction back to original image size
    pred_mask_resized = cv2.resize(pred_mask, (image_rgb.shape[1], image_rgb.shape[0]))
    binary_mask = (pred_mask_resized > 0.5).astype(np.uint8)

    # 5. Create overlay mask in green
    overlay = image_rgb.copy()
    overlay[binary_mask == 1] = [0, 255, 0]  # Green where mask is 1

    blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

    # 6. Show images side by side
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(blended)
    plt.title("Prediction Overlay")
    plt.axis("off")
    plt.show()