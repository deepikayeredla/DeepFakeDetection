# # Face extraction and preprocessing utilities
# import os
# import cv2
# import torch
# from facenet_pytorch import MTCNN
# from PIL import Image
# from tqdm import tqdm
# import pandas as pd

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# mtcnn = MTCNN(keep_all=False, device=device)

# def extract_faces_from_video(video_path, out_dir, max_frames=None):
#     """
#     Extract faces from a video and save as images.
#     Args:
#         video_path (str): Path to video file
#         out_dir (str): Directory to save cropped faces
#         max_frames (int, optional): Max frames to process
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     vid = cv2.VideoCapture(video_path)
#     idx = 0
#     saved = 0

#     while True:
#         ret, frame = vid.read()
#         if not ret or (max_frames and saved >= max_frames):
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         box, _ = mtcnn.detect(rgb_frame)

#         if box is not None:
#             face = mtcnn.extract(rgb_frame, [box[0]])[0]  # PIL Image
#             face.save(os.path.join(out_dir, f'{idx:06d}.jpg'))
#             saved += 1

#         idx += 1

#     vid.release()
#     print(f"Saved {saved} faces from {video_path}")

# def create_metadata_csv(processed_dir, label, csv_path):
#     """
#     Create a CSV file for the processed frames with labels
#     Args:
#         processed_dir (str): Directory containing face images
#         label (int): 0 for real, 1 for fake
#         csv_path (str): Path to save CSV
#     """
#     rows = []
#     for fname in os.listdir(processed_dir):
#         if fname.endswith('.jpg'):
#             rows.append({
#                 'file_path': os.path.join(processed_dir, fname),
#                 'label': label
#             })
#     df = pd.DataFrame(rows)
#     df.to_csv(csv_path, index=False)
#     print(f"Metadata saved to {csv_path}")


# Face extraction, preprocessing, and DataLoader utilities
import os
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Device for MTCNN / training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)

# --------------------------
# Video face extraction
# --------------------------


def extract_faces_from_video(video_path, out_dir, max_frames=None):
    """
    Extract faces from a video and save as images.
    Args:
        video_path (str): Path to video file
        out_dir (str): Directory to save cropped faces
        max_frames (int, optional): Max frames to process
    """
    os.makedirs(out_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0

    while True:
        ret, frame = vid.read()
        if not ret or (max_frames and saved >= max_frames):
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box, _ = mtcnn.detect(rgb_frame)

        if box is not None:
            face = mtcnn.extract(rgb_frame, [box[0]])[0]  # PIL Image
            face.save(os.path.join(out_dir, f'{idx:06d}.jpg'))
            saved += 1

        idx += 1

    vid.release()
    print(f"Saved {saved} faces from {video_path}")


# --------------------------
# Metadata CSV creation
# --------------------------
def create_metadata_csv(processed_dir, label, csv_path):
    """
    Create a CSV file for the processed frames with labels
    Args:
        processed_dir (str): Directory containing face images
        label (int): 0 for real, 1 for fake
        csv_path (str): Path to save CSV
    """
    rows = []
    for root, _, files in os.walk(processed_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    'file_path': os.path.join(root, fname),
                    'label': label
                })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")


# --------------------------
# Dataset & DataLoader for images
# --------------------------
class FaceFrameDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['file_path']).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return img, label


def get_dataloader(csv_file, batch_size=16, shuffle=True):
    dataset = FaceFrameDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
