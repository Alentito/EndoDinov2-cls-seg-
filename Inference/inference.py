import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
import gradio as gr
import matplotlib.pyplot as plt
import json

# ---------------- CONFIG ----------------
WEIGHTS = "Transfer_learned_final.pth"
UNET_WEIGHTS = "unet_baseline_best_Final.pth"
IMAGE_SIZE = 224
UNET_IMAGE_SIZE = 320

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3', 'cecum',
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a', 'esophagitis-b-d',
    'hemorrhoids', 'ileum', 'impacted-stool', 'polyps', 'pylorus', 'retroflex-rectum',
    'retroflex-stomach', 'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-1',
    'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2', 'ulcerative-colitis-grade-2-3',
    'ulcerative-colitis-grade-3', 'z-line'
]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

unet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((UNET_IMAGE_SIZE, UNET_IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ---------------- MODEL DEFINITIONS ----------------
class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        target = self.model.backbone.norm
        target.register_forward_hook(self._forward_hook)
        target.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self):
        patch_acts = self.activations[:, 1:, :]
        cls_grad = self.gradients[:, 0, :]
        weights = cls_grad.unsqueeze(1)
        cam = (weights * patch_acts).sum(dim=2)
        cam = torch.relu(cam)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max + 1e-6)
        return cam

class DinoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
        self.head = nn.Linear(384, num_classes)
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                feat = out["x_norm_clstoken"]
            elif "x_norm_patchtokens" in out:
                feat = out["x_norm_patchtokens"].mean(dim=1)
            else:
                raise RuntimeError("Unknown DINOv2 output format")
        elif out.dim() == 3:
            feat = out.mean(dim=1)
        elif out.dim() == 2:
            feat = out
        else:
            raise RuntimeError(f"Unexpected output shape: {out.shape}")
        return self.head(feat)

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, num_classes):
        model = cls(num_classes)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        model.head.load_state_dict(ckpt["head_state_dict"], strict=True)
        return model

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# ---------------- INFERENCE FUNCTION ----------------
def inference(image):
    # Convert to RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = transform(rgb).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    # Load classifier
    model = DinoClassifier.load_from_checkpoint(WEIGHTS, num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.eval()
    cam_extractor = ViTGradCAM(model)

    # Classification
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred_idx = logits.argmax(dim=1).item()
    label = CLASS_NAMES[pred_idx]
    conf = float(probs[0, pred_idx])

    # Grad-CAM
    model.zero_grad()
    logits[0, pred_idx].backward()
    cam_tokens = cam_extractor.generate()
    num_patches = cam_tokens.shape[1]
    grid_size = int(num_patches ** 0.5)
    cam_map = cam_tokens[0].reshape(grid_size, grid_size)
    cam_map = cam_map - cam_map.min()
    cam_map = cam_map / (cam_map.max() + 1e-8)
    cam_map = cam_map.detach().cpu().numpy()
    cam_map = cv2.resize(cam_map, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = np.uint8(255 * cam_map)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (bgr.shape[1], bgr.shape[0]))
    overlay = cv2.addWeighted(bgr, 0.6, heatmap_color, 0.4, 0)
    cv2.putText(overlay, f"{label} ({conf:.2f})", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Prepare output images
    results = {
        "classification": {
            "label": label,
            "confidence": conf
        },
        "heatmap": overlay_rgb
    }

    # If polyp, run segmentation
    if label == "polyps":
        # Load UNet
        unet_model = UNet().to(DEVICE)
        unet_model.load_state_dict(torch.load(UNET_WEIGHTS, map_location=DEVICE))
        unet_model.eval()
        img_unet = cv2.resize(rgb, (UNET_IMAGE_SIZE, UNET_IMAGE_SIZE))
        img_tensor = torch.from_numpy(img_unet).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = unet_model(img_tensor)
            mask_pred = (torch.sigmoid(output) > 0.5).float()
        mask_np = mask_pred.squeeze().cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_unet = img_unet.copy()  # already RGB
        cv2.drawContours(overlay_unet, contours, -1, (255, 255, 255), 2)  # draw white contour
        overlay_unet_rgb = overlay_unet
        results["segmentation"] = {
            "mask": mask_np,
            "overlay": overlay_unet_rgb
        }
        # Encode image to PNG bytes in memory
        _, img_encoded = cv2.imencode('.png', overlay_unet_rgb)
        segmentation_bytes = img_encoded.tobytes()

    # Structured JSON output
    findings = {
        "predicted_class": label,
        "confidence": conf,
        "polyp_detected": label == "polyps",
        "segmentation_available": "segmentation" in results,
        "key_findings": []
    }
    if label == "polyps":
        findings["key_findings"].append("Polyp detected and segmented.")
    else:
        findings["key_findings"].append(f"Classified as {label}.")

    # Import and call LM Studio report generator
    from lmstudio_report import get_lmstudio_report
    report = get_lmstudio_report(findings, image_bytes=segmentation_bytes if label == "polyps" else None)

    return (
        overlay_rgb,
        results.get("segmentation", {}).get("overlay", None),
        json.dumps(findings, indent=2),
        report
    )

# ---------------- GRADIO GUI ----------------
with gr.Blocks(title="Endoscopy Classifier & Polyp Segmentation") as iface:
    gr.Markdown("# Endoscopy Classifier & Polyp Segmentation")#\nDrop an image to classify, visualize GradCAM, segment polyps if detected, and generate a medical report using LM Studio.
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Drop an Endoscopy Image")
        with gr.Column(scale=1):
            with gr.Row():
                class_img = gr.Image(type="numpy", label="Classification + GradCAM")
                seg_img = gr.Image(type="numpy", label="Polyp Segmentation Overlay (if polyp)")
            findings_json = gr.JSON(label="Key Findings (JSON)")
            report_box = gr.Textbox(label="LM Studio Medical Report", lines=12)
    input_img.change(
        inference,
        inputs=[input_img],
        outputs=[class_img, seg_img, findings_json, report_box]
    )

if __name__ == "__main__":
    iface.launch()