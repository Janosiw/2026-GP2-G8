import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import os
import cv2

CLASSIFIER_MODEL_PATH = "class_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EfficientNetB1_Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b1",
            pretrained=False
        )
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

        reduced = [dr(f) for f, dr in zip(feats, self.dim_reduce)]
        aligned = [sr(r) for r, sr in zip(reduced, self.spatial_reduce)]
        fused = torch.cat(aligned, dim=1)
        z = F.adaptive_avg_pool2d(fused, (1, 1)).view(fused.size(0), -1)
        z = self.ln(z)
        z = self.fc(z)
        z = self.relu(z)
        z = self.bn(z)
        logits = self.out(z)
        return logits


def load_classifier_model():
    model = EfficientNetB1_Classifier(num_classes=4)
    state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(" Model Loaded Successfully — (Correct Architecture)")
    return model


def preprocess_img(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)


def generate_gradcam(model, img_path, save_name="gradcam.png"):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    target_layer = model.model.conv_head
    conv_features = []
    conv_grads = []

    def forward_hook(module, inp, output):
        conv_features.append(output)

    def backward_hook(module, grad_in, grad_out):
        conv_grads.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    logits = model(input_tensor)
    pred_class = logits.argmax(1).item()

    model.zero_grad()
    logits[0, pred_class].backward()

    fmap = conv_features[0].detach().cpu().numpy()[0]
    grads = conv_grads[0].detach().cpu().numpy()[0]
    weights = grads.mean(axis=(1, 2))
    cam = np.maximum(np.sum(weights[:, None, None] * fmap, axis=0), 0)
    cam /= (cam.max() + 1e-8)
    cam = cv2.resize(cam, img.size)

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

    out_dir = "static/uploads/gradcam"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, save_name)
    cv2.imwrite(save_path, overlay)

    fh.remove()
    bh.remove()

    return save_path, pred_class


CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def classify_image(model, img_path):
    x = preprocess_img(img_path).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    tumor_type = CLASSES[idx]
    confidence = float(probs[idx]) * 100.0
    return tumor_type, confidence
