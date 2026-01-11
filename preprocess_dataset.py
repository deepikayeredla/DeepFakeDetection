# # # import os
# # # from utils.data_utils import extract_faces_from_video
# # # import pandas as pd

# # # # Paths
# # # RAW_DIR = "data/raw"
# # # PROCESSED_DIR = "data/processed"
# # # METADATA_CSV = "data/metadata.csv"

# # # os.makedirs(PROCESSED_DIR, exist_ok=True)

# # # metadata_rows = []

# # # # Helper function to process a folder of videos
# # # def process_videos(video_folder, label):
# # #     folder_path = os.path.join(RAW_DIR, video_folder)
# # #     for vid_name in os.listdir(folder_path):
# # #         if not vid_name.endswith((".mp4", ".mov", ".avi")):
# # #             continue
# # #         video_path = os.path.join(folder_path, vid_name)
# # #         out_dir = os.path.join(PROCESSED_DIR, video_folder, vid_name.split(".")[0])
# # #         os.makedirs(out_dir, exist_ok=True)

# # #         print(f"Processing {video_path} ...")
# # #         extract_faces_from_video(video_path, out_dir)

# # #         # Add each extracted frame to metadata
# # #         for fname in os.listdir(out_dir):
# # #             if fname.endswith(".jpg"):
# # #                 metadata_rows.append({
# # #                     "file_path": os.path.join(out_dir, fname),
# # #                     "label": label
# # #                 })

# # # # Process real videos (label=0)
# # # process_videos("real", label=0)

# # # # Process fake videos (label=1)
# # # process_videos("fake", label=1)

# # # # Save metadata.csv
# # # df = pd.DataFrame(metadata_rows)
# # # df.to_csv(METADATA_CSV, index=False)
# # # print(f"Metadata CSV created at {METADATA_CSV}")
# # # print(f"Total frames: {len(metadata_rows)}")


# # import os
# # import pandas as pd

# # RAW_DIR = "data/raw"
# # METADATA_CSV = "data/metadata.csv"

# # metadata_rows = []

# # # Process real images
# # for fname in os.listdir(os.path.join(RAW_DIR, "real")):
# #     if fname.endswith((".jpg", ".png")):
# #         metadata_rows.append({
# #             "file_path": os.path.join(RAW_DIR, "real", fname),
# #             "label": 0
# #         })

# # # Process fake images
# # for fname in os.listdir(os.path.join(RAW_DIR, "fake")):
# #     if fname.endswith((".jpg", ".png")):
# #         metadata_rows.append({
# #             "file_path": os.path.join(RAW_DIR, "fake", fname),
# #             "label": 1
# #         })

# # # Save metadata.csv
# # df = pd.DataFrame(metadata_rows)
# # df.to_csv(METADATA_CSV, index=False)
# # print(f"Metadata CSV created at {METADATA_CSV}")
# # print(f"Total images: {len(metadata_rows)}")

# import os
# import shutil
# import pandas as pd

# RAW_DIR = "data/raw"
# PROCESSED_DIR = "data/processed"
# METADATA_CSV = "data/metadata.csv"

# os.makedirs(PROCESSED_DIR, exist_ok=True)
# metadata_rows = []

# for category in ["real", "fake"]:
#     src_dir = os.path.join(RAW_DIR, category)
#     dest_dir = os.path.join(PROCESSED_DIR, category)
#     os.makedirs(dest_dir, exist_ok=True)

#     for fname in os.listdir(src_dir):
#         if fname.endswith((".jpg", ".png")):
#             src_path = os.path.join(src_dir, fname)
#             dest_path = os.path.join(dest_dir, fname)
#             shutil.copy(src_path, dest_path)  # copy image to processed/
#             label = 0 if category == "real" else 1
#             metadata_rows.append({
#                 "file_path": dest_path,
#                 "label": label
#             })

# # Save metadata CSV
# df = pd.DataFrame(metadata_rows)
# df.to_csv(METADATA_CSV, index=False)
# print(f"Metadata CSV created at {METADATA_CSV}")
# print(f"Total images: {len(metadata_rows)}")


import os
import shutil
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
METADATA_CSV = "data/metadata.csv"

os.makedirs(PROCESSED_DIR, exist_ok=True)
metadata_rows = []

for category in ["real", "fake"]:
    src_dir = os.path.join(RAW_DIR, category)

    if not os.path.exists(src_dir):
        print(f"Folder not found: {src_dir}")
        continue  # skip if folder missing

    dest_dir = os.path.join(PROCESSED_DIR, category)
    os.makedirs(dest_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            src_path = os.path.join(src_dir, fname)
            dest_path = os.path.join(dest_dir, fname)
            shutil.copy(src_path, dest_path)
            label = 0 if category == "real" else 1
            metadata_rows.append({
                "file_path": dest_path,
                "label": label
            })

# Save metadata CSV
df = pd.DataFrame(metadata_rows)
df.to_csv(METADATA_CSV, index=False)
print(f"Metadata CSV created at {METADATA_CSV}")
print(f"Total images: {len(metadata_rows)}")
