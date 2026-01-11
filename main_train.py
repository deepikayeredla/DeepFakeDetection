# # Main script to train model
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from utils.data_utils import get_dataloader
# from models.baseline_cnn import FrameCNN
# from utils.train_utils import train_one_epoch, validate

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load dataloaders
# train_loader = get_dataloader("data/metadata.csv", batch_size=16, shuffle=True)
# val_loader = get_dataloader("data/metadata.csv", batch_size=16, shuffle=False)

# # Initialize model
# model = FrameCNN(pretrained=True).to(device)

# # Optimizer & loss
# optimizer = Adam(model.parameters(), lr=1e-4)
# criterion = nn.BCELoss()

# # Training loop
# num_epochs = 5
# for epoch in range(num_epochs):
#     train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
#     val_loss, val_auc = validate(model, val_loader, criterion, device)

#     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

#     # Save checkpoint
#     torch.save(model.state_dict(), f"checkpoints/frame_cnn_epoch{epoch+1}.pth")


# Main script to train model and save metrics
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.data_utils import get_dataloader
from models.baseline_cnn import FrameCNN
from utils.train_utils import train_one_epoch, validate
import pandas as pd
# ========== Use Google Drive paths ==========
base_path = "/content/drive/MyDrive/DeepFakeDetection"
data_path = f"{base_path}/data/metadata.csv"
checkpoint_path = f"{base_path}/checkpoints/"
output_path = f"{base_path}/outputs/"


# Create folders if not exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataloaders
train_loader = get_dataloader("data/metadata.csv", batch_size=16, shuffle=True)
val_loader = get_dataloader("data/metadata.csv", batch_size=16, shuffle=False)

# Initialize model
model = FrameCNN(pretrained=True).to(device)

# Optimizer & loss
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Lists to store metrics
train_losses, val_losses = [], []
train_aucs, val_aucs = [], []

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_auc = train_one_epoch(
        model, train_loader, optimizer, criterion, device)
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
          f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/frame_cnn_epoch{epoch+1}.pth")

# Save metrics to CSV for plotting
df_metrics = pd.DataFrame({
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_auc': train_aucs,
    'val_auc': val_aucs
})
df_metrics.to_csv('outputs/training_metrics.csv', index=False)
print("Training metrics saved to outputs/training_metrics.csv")
