# # import os
# # import torch
# # from PIL import Image
# # from torchvision import transforms
# # from torch.utils.data import Dataset, DataLoader
# # from models.baseline_cnn import FrameCNN

# # # ===============================
# # # Configuration
# # # ===============================
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # base_path = "data/processed"  # Update to your dataset path
# # metadata_csv = os.path.join(base_path, "metadata.csv")
# # checkpoint_path = "checkpoints/frame_cnn_epoch5.pth"

# # # ===============================
# # # Dataset
# # # ===============================


# # class FaceFrameDataset(Dataset):
# #     def __init__(self, csv_file):
# #         import pandas as pd
# #         self.df = pd.read_csv(csv_file)
# #         self.transform = transforms.Compose([
# #             transforms.Resize((224, 224)),
# #             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                                  std=[0.229, 0.224, 0.225])
# #         ])

# #     def __len__(self):
# #         return len(self.df)

# #     def __getitem__(self, idx):
# #         row = self.df.iloc[idx]
# #         img = Image.open(row['file_path']).convert('RGB')
# #         img = self.transform(img)
# #         label = torch.tensor(row['label'], dtype=torch.float32)
# #         return img, label


# # # ===============================
# # # Load Model
# # # ===============================
# # if not os.path.exists(checkpoint_path):
# #     raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# # model = FrameCNN(pretrained=False).to(device)
# # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# # model.eval()

# # # ===============================
# # # Load Dataset
# # # ===============================
# # dataset = FaceFrameDataset(metadata_csv)
# # loader = DataLoader(dataset, batch_size=16, shuffle=False)

# # # ===============================
# # # Evaluation
# # # ===============================
# # correct = 0
# # total = 0

# # with torch.no_grad():
# #     for images, labels in loader:
# #         images = images.to(device)
# #         labels = labels.to(device)
# #         outputs = model(images)
# #         # Convert logits to probabilities
# #         probs = torch.sigmoid(outputs).squeeze()
# #         preds = (probs >= 0.5).float()
# #         correct += (preds == labels).sum().item()
# #         total += labels.size(0)

# # accuracy = correct / total if total > 0 else 0
# # print(f"✅ Overall Accuracy: {accuracy*100:.2f}% ({correct}/{total})")


# import os
# import torch
# import pandas as pd
# from torch.utils.data import DataLoader
# from models.baseline_cnn import FrameCNN
# from utils.data_utils import FaceFrameDataset  # your dataset class

# # ======== Paths ========
# base_path = r"C:\Users\Deepika\Downloads\DeepFakeDetectorr\DeepFakeDetector"
# metadata_csv = os.path.join(base_path, "data",  "metadata.csv")
# checkpoint_path = os.path.join(
#     base_path, "checkpoints", "frame_cnn_epoch5.pth")

# # ======== Check files ========
# if not os.path.exists(metadata_csv):
#     raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
# if not os.path.exists(checkpoint_path):
#     raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

# print(f"[INFO] Metadata CSV found: {metadata_csv}")
# print(f"[INFO] Checkpoint found: {checkpoint_path}")

# # ======== Device ========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[INFO] Using device: {device}")

# # ======== Load model ========
# model = FrameCNN(pretrained=False).to(device)
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# model.eval()
# print("[INFO] Model loaded successfully.")

# # ======== Prepare dataset ========
# dataset = FaceFrameDataset(metadata_csv)
# loader = DataLoader(dataset, batch_size=16, shuffle=False)
# print(f"[INFO] Dataset ready: {len(dataset)} samples.")

# # ======== Evaluate ========
# correct = 0
# total = 0

# with torch.no_grad():
#     for imgs, labels in loader:
#         imgs = imgs.to(device)
#         labels = labels.to(device)
#         outputs = model(imgs)
#         probs = torch.sigmoid(outputs).squeeze()
#         preds = (probs >= 0.6).float()
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

# accuracy = correct / total * 100
# print(f"[RESULT] Overall Accuracy: {accuracy:.2f}%")


import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from models.baseline_cnn import FrameCNN
from utils.data_utils import FaceFrameDataset  # your dataset class

# ======== Paths ========
base_path = r"C:\Users\Deepika\Downloads\DeepFakeDetectorr\DeepFakeDetector"
metadata_csv = os.path.join(base_path, "data", "processed", "metadata.csv")
checkpoint_path = os.path.join(
    base_path, "checkpoints", "frame_cnn_epoch5.pth")

# ======== Verify working directory ========
print(f"[DEBUG] Current working directory: {os.getcwd()}")
os.chdir(base_path)  # Ensures correct relative imports
print(f"[INFO] Changed working directory to: {os.getcwd()}")

# ======== Check files ========
if not os.path.exists(metadata_csv):
    raise FileNotFoundError(f"[ERROR] Metadata CSV not found: {metadata_csv}")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"[ERROR] Model checkpoint not found: {checkpoint_path}")

print(f"[INFO] Metadata CSV found: {metadata_csv}")
print(f"[INFO] Checkpoint found: {checkpoint_path}")

# ======== Device ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ======== Load model ========
model = FrameCNN(pretrained=False).to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("[INFO] Model loaded successfully.")

# ======== Prepare dataset ========
# Validate metadata paths before loading
df = pd.read_csv(metadata_csv)
if not os.path.exists(df.iloc[0]["file_path"]):
    print("[WARN] Sample path in metadata not found — verifying paths...")
    # Fix common issue if paths contain 'data/processed'
    df["file_path"] = df["file_path"].str.replace(
        "data/processed", "data", regex=False)
    df.to_csv(metadata_csv, index=False)
    print("[INFO] Metadata file paths corrected automatically.")

dataset = FaceFrameDataset(metadata_csv)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
print(f"[INFO] Dataset ready: {len(dataset)} samples.")

# ======== Evaluate ========
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs >= 0.6).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"[RESULT] Overall Accuracy: {accuracy:.2f}%")
