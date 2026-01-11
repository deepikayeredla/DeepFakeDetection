import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# ====================================================
# 1️⃣ Image Preprocessing
# ====================================================

def preprocess_image(image_path, device):
    """
    Loads and preprocesses an image for model inference.
    Converts to tensor and normalizes to match training conditions.
    Returns: (input_tensor, original_PIL_image)
    """
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize same as training input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor, image


# ====================================================
# 2️⃣ Simple Grad-CAM Implementation
# ====================================================

def generate_gradcam_simple(model, target_layer, input_tensor):
    """
    Simple Grad-CAM for models with a single scalar output (e.g., sigmoid).
    Works with binary classifiers like FrameCNN.
    """
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    # Forward
    output = model(input_tensor)
    print("[INFO] Predicted class:", int(output.sigmoid().item() > 0.5))

    # Backward (for single output)
    model.zero_grad()
    score = output  # single scalar
    score.backward()

    # Remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    # Extract activations and gradients
    act = activations[0].detach()
    grad = gradients[0].detach()

    # Compute CAM
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Normalize
    cam = cam.squeeze().cpu().numpy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam


# ====================================================
# 3️⃣ Visualization Function
# ====================================================

def generate_gradcam_visualization(model, target_layer, input_tensor, original_image, save_path=None):
    """
    Generates Grad-CAM visualization and optionally saves it.
    """
    print("[INFO] Generating Grad-CAM heatmap...")
    cam = generate_gradcam_simple(model, target_layer, input_tensor)

    # Resize to match original image
    cam = cv2.resize(cam, (original_image.width, original_image.height))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_np = np.array(original_image)
    overlay = np.float32(heatmap) / 255 + np.float32(original_np) / 255
    overlay /= overlay.max()

    vis = np.uint8(255 * overlay)

    # Save visualization
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"✅ Grad-CAM visualization saved to {save_path}")

    # Display for confirmation (optional)
    plt.imshow(vis)
    plt.axis('off')
    plt.show()

    return vis
