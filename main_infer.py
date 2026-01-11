# # import torch
# # import os
# # from PIL import Image
# # from models.baseline_cnn import FrameCNN
# # from utils.explainability import preprocess_image, generate_gradcam_visualization

# # # ============================
# # # Device Setup
# # # ============================
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"[INFO] Using device: {device}")

# # # ============================
# # # Load Trained Model
# # # ============================
# # model = FrameCNN(pretrained=False).to(device)
# # checkpoint_path = "checkpoints/frame_cnn_epoch5.pth"

# # if not os.path.exists(checkpoint_path):
# #     raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

# # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# # model.eval()
# # print("✅ Model loaded successfully!")

# # # ============================
# # # Target Layer for Grad-CAM
# # # ============================
# # try:
# #     target_layer = model.backbone.conv_head  # last conv layer
# #     print("[INFO] Target layer for Grad-CAM: backbone.conv_head")
# # except AttributeError:
# #     raise ValueError(
# #         "❌ Could not find conv_head in backbone. Inspect `model.backbone`.")

# # # ============================
# # # Inference Function
# # # ============================


# # def infer_image(image_path, save_gradcam=False, save_path="outputs/visualizations/gradcam.jpg"):
# #     """
# #     Performs prediction and optionally saves Grad-CAM visualization.
# #     """
# #     if not os.path.exists(image_path):
# #         raise FileNotFoundError(f"❌ Image not found: {image_path}")

# #     print(f"\n[INFO] Loading image: {image_path}")
# #     input_tensor, orig_image = preprocess_image(image_path, device=device)

# #     with torch.no_grad():
# #         output = model(input_tensor)
# #         prob = torch.sigmoid(output).item()  # convert logits to probability

# #     label = "FAKE" if prob > 0.53 else "REAL"
# #     print(f"\nPrediction: {label} ({prob:.4f})")

# #     # Save Grad-CAM visualization
# #     if save_gradcam:
# #         print("[INFO] Generating Grad-CAM visualization...")
# #         # vis = generate_gradcam_visualization(
# #         #     model, target_layer, image_path, device=device)
# #         vis = generate_gradcam_visualization(
# #             model, target_layer, input_tensor, orig_image, save_path=save_path)

# #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
# #         Image.fromarray(vis).save(save_path)
# #         print(f"✅ Grad-CAM visualization saved to {save_path}")


# # # ============================
# # # Run Example
# # # ============================
# # if __name__ == "__main__":
# #     # ✅ Change this only when testing new images
# #     image_path = "Test/Fake/fake_64.jpg"
# #     infer_image(image_path, save_gradcam=True)


# import torch
# import os
# from PIL import Image, ImageDraw, ImageFont
# from models.baseline_cnn import FrameCNN
# from utils.explainability import preprocess_image, generate_gradcam_visualization

# # ============================
# # Device Setup
# # ============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[INFO] Using device: {device}")

# # ============================
# # Load Trained Model
# # ============================
# model = FrameCNN(pretrained=False).to(device)
# checkpoint_path = "checkpoints/frame_cnn_epoch5.pth"

# if not os.path.exists(checkpoint_path):
#     raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# model.eval()
# print("✅ Model loaded successfully!")

# # ============================
# # Target Layer for Grad-CAM
# # ============================
# try:
#     target_layer = model.backbone.conv_head  # last conv layer
#     print("[INFO] Target layer for Grad-CAM: backbone.conv_head")
# except AttributeError:
#     raise ValueError(
#         "❌ Could not find conv_head in backbone. Inspect `model.backbone`."
#     )


# # ============================
# # Inference Function
# # ============================
# def infer_image(image_path, save_gradcam=False, save_path="outputs/visualizations/gradcam.jpg"):
#     """
#     Performs prediction and optionally saves Grad-CAM visualization
#     with label overlay.
#     """
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"❌ Image not found: {image_path}")

#     print(f"\n[INFO] Loading image: {image_path}")
#     input_tensor, orig_image = preprocess_image(image_path, device=device)

#     with torch.no_grad():
#         output = model(input_tensor)
#         prob = torch.sigmoid(output).item()  # convert logits to probability

#     label = "FAKE" if prob > 0.53 else "REAL"
#     print(f"\nPrediction: {label} ({prob:.4f})")

#     if save_gradcam:
#         print("[INFO] Generating Grad-CAM visualization...")
#         vis = generate_gradcam_visualization(
#             model, target_layer, input_tensor, orig_image, save_path=None
#         )

#         # Convert to PIL for adding label text
#         vis_pil = Image.fromarray(vis)
#         draw = ImageDraw.Draw(vis_pil)

#         # Use default PIL font; can replace with custom TTF if desired
#         try:
#             font = ImageFont.truetype("arial.ttf", size=30)
#         except:
#             font = ImageFont.load_default()

#         text = f"{label} ({prob:.2f})"
#         text_width, text_height = draw.textsize(text, font=font)

#         # Position: bottom-center
#         x = (vis_pil.width - text_width) // 2
#         y = vis_pil.height - text_height - 10

#         # Draw semi-transparent rectangle for readability
#         rect_padding = 10
#         draw.rectangle(
#             [x - rect_padding, y - rect_padding, x + text_width +
#                 rect_padding, y + text_height + rect_padding],
#             fill=(0, 0, 0, 128)
#         )

#         # Draw text
#         draw.text((x, y), text, fill=(255, 255, 255), font=font)

#         # Save final image
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         vis_pil.save(save_path)
#         print(f"✅ Grad-CAM visualization with label saved to {save_path}")


# # ============================
# # Run Example
# # ============================
# if __name__ == "__main__":
#     image_path = "Test/Fake/fake_64.jpg"  # Update when testing new images
#     infer_image(image_path, save_gradcam=True)


import torch
import os
from PIL import Image
from models.baseline_cnn import FrameCNN
from utils.explainability import preprocess_image, generate_gradcam_simple
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ============================
# Device Setup
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ============================
# Load Trained Model
# ============================
model = FrameCNN(pretrained=False).to(device)
checkpoint_path = "checkpoints/frame_cnn_epoch5.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# ============================
# Target Layer for Grad-CAM
# ============================
try:
    target_layer = model.backbone.conv_head  # last conv layer
    print("[INFO] Target layer for Grad-CAM: backbone.conv_head")
except AttributeError:
    raise ValueError(
        "❌ Could not find conv_head in backbone. Inspect `model.backbone`."
    )

# ============================
# Grad-CAM Visualization
# ============================


def generate_gradcam_visualization(model, target_layer, input_tensor, original_image, label="UNKNOWN", save_path=None):
    """
    Generates Grad-CAM visualization and overlays prediction label.
    """
    print("[INFO] Generating Grad-CAM heatmap...")
    cam = generate_gradcam_simple(model, target_layer, input_tensor)

    # Resize CAM to match original image
    cam = cv2.resize(cam, (original_image.width, original_image.height))

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_np = np.array(original_image)
    overlay = np.float32(heatmap) / 255 + np.float32(original_np) / 255
    overlay /= overlay.max()
    vis = np.uint8(255 * overlay)

    # Add label text at the bottom
    text = f"Prediction: {label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int((vis.shape[1] - text_size[0]) / 2)  # center horizontally
    text_y = vis.shape[0] - 10  # 10 px from bottom
    cv2.putText(vis, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"✅ Grad-CAM visualization saved to {save_path}")

    # Display
    plt.imshow(vis)
    plt.axis('off')
    plt.show()

    return vis

# ============================
# Inference Function
# ============================


def infer_image(image_path, save_gradcam=False, save_path="outputs/visualizations/gradcam.jpg"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    print(f"\n[INFO] Loading image: {image_path}")
    input_tensor, orig_image = preprocess_image(image_path, device=device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()  # probability between 0 and 1

    label = "FAKE" if prob > 0.53 else "REAL"
    print(f"\nPrediction: {label} ({prob:.4f})")

    if save_gradcam:
        vis = generate_gradcam_visualization(
            model, target_layer, input_tensor, orig_image, label=label, save_path=save_path
        )


# ============================
# Run Example
# ============================
if __name__ == "__main__":
    # ✅ Change only when tes
    # ting new images
    image_path = "Test/Fake/2.jpg"
    infer_image(image_path, save_gradcam=True)
