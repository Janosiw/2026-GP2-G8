import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import segmentation_models_pytorch as smp

SEG_MODEL_PATH = "SEG_Model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_segmentation_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b1",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_use_batchnorm=True,
        decoder_attention_type="scse"
    )

    state_dict = torch.load(SEG_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    print("U-Net++ Loaded Successfully")
    return model


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0],
                             std=[1.0, 1.0, 1.0])
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor, img.size


def segment_image(model, img_path, scan_id=None):
    model.eval()
    img_tensor, original_size = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        mask_pred = torch.sigmoid(model(img_tensor)).cpu().numpy()[0, 0]

    mask = (mask_pred > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    mask_dir = "static/uploads/masks"
    os.makedirs(mask_dir, exist_ok=True)
    mask_filename = f"mask_{scan_id}.png" if scan_id else f"mask_{os.path.basename(img_path)}"
    mask_path = os.path.join(mask_dir, mask_filename)

    cv2.imwrite(mask_path, mask_resized)
    print(f"Saved mask at: {mask_path}")
    return mask_path
