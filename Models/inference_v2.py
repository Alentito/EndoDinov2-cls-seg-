import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
import gradio as gr
import json

# ---------------- CONFIG ----------------
WEIGHTS = "Transfer_learned_final.pth"
UNET_WEIGHTS = "unet_baseline_best_Final.pth"
IMAGE_SIZE = 224
UNET_IMAGE_SIZE = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_VIDEO_SECONDS = 60
FPS_SAMPLE = 5  # sample 5 frames per second

CLASS_NAMES = [
    'barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3', 'cecum',
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a', 'esophagitis-b-d',
    'hemorrhoids', 'ileum', 'impacted-stool', 'polyps', 'pylorus', 'retroflex-rectum',
    'retroflex-stomach', 'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-1',
    'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2', 'ulcerative-colitis-grade-2-3',
    'ulcerative-colitis-grade-3', 'z-line'
]

# ---------------- TRANSFORMS ----------------
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

#---------------- IMPORT MODELS ----------------



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



# ---------------- MODELS ----------------
# Use your existing DinoClassifier and UNet definitions
# Assuming ViTGradCAM is also defined as in your previous code

# Load models once
dino_model = DinoClassifier.load_from_checkpoint(WEIGHTS, num_classes=len(CLASS_NAMES)).to(DEVICE)
dino_model.eval()
cam_extractor = ViTGradCAM(dino_model)

unet_model = UNet().to(DEVICE)
unet_model.load_state_dict(torch.load(UNET_WEIGHTS, map_location=DEVICE))
unet_model.eval()

# ---------------- TEMPORAL SMOOTHING ----------------
def smooth_predictions(pred_list, alpha=0.3):
    """Exponential moving average smoothing for classification probabilities."""
    if len(pred_list) == 0:
        return []
    smoothed = [pred_list[0]]
    for p in pred_list[1:]:
        smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    return smoothed

def frames_to_video(frames, output_path, fps=5):
    if frames is None or len(frames) == 0:
        return None
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    return output_path



# ---------------- VIDEO INFERENCE ----------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), MAX_VIDEO_SECONDS * fps))
    step = max(1, int(fps / FPS_SAMPLE))
    
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % step != 0:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Run inference on frames
    class_probs_list = []
    heatmaps = []
    seg_overlays = []
    frame_outputs = []
    
    for rgb in frames:
        x = transform(rgb).unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)
        # Classification
        logits = dino_model(x)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        class_probs_list.append(probs)
        pred_idx = logits.argmax(dim=1).item()
        label = CLASS_NAMES[pred_idx]
        
        # Grad-CAM
        dino_model.zero_grad()
        logits[0, pred_idx].backward()
        cam_tokens = cam_extractor.generate()
        num_patches = cam_tokens.shape[1]
        grid_size = int(num_patches ** 0.5)
        cam_map = cam_tokens[0].reshape(grid_size, grid_size)
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        cam_map = cv2.resize(cam_map.detach().cpu().numpy(), (rgb.shape[1], rgb.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(rgb, 0.6, heatmap_color, 0.4, 0)
        cv2.putText(overlay, f"{label} ({probs[pred_idx]:.2f})", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        heatmaps.append(overlay)
        
        # Segmentation if polyp
        if label == "polyps":
            img_unet = cv2.resize(rgb, (UNET_IMAGE_SIZE, UNET_IMAGE_SIZE))
            img_tensor = torch.from_numpy(img_unet).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = unet_model(img_tensor)
                mask_pred = (torch.sigmoid(output) > 0.5).float()
            mask_np = mask_pred.squeeze().cpu().numpy().astype(np.uint8)

            # Resize mask back to original frame size
            mask_resized = cv2.resize(
                mask_np,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            contours, _ = cv2.findContours(
                mask_resized,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            overlay_unet = rgb.copy()
            cv2.drawContours(overlay_unet, contours, -1, (255, 255, 255), 2)

            seg_overlays.append(overlay_unet)
        else:
            seg_overlays.append(None)
    
    # Apply temporal smoothing to classification
    class_probs_smoothed = smooth_predictions(class_probs_list)
    
    # Construct JSON report per frame
    findings = []
    for i, probs in enumerate(class_probs_smoothed):
        pred_idx = np.argmax(probs)
        label = CLASS_NAMES[pred_idx]
        report = {
            "frame_idx": i,
            "predicted_class": label,
            "confidence": float(probs[pred_idx]),
            "polyp_detected": label == "polyps",
            "segmentation_available": seg_overlays[i] is not None
        }
        findings.append(report)
    
    heatmap_video_path = "temp_heatmap.mp4"
    seg_video_path = "temp_segmentation.mp4"

    heatmap_video_file = frames_to_video(heatmaps, heatmap_video_path, fps=FPS_SAMPLE)
    seg_video_file = frames_to_video([f for f in seg_overlays if f is not None], seg_video_path, fps=FPS_SAMPLE)

    return heatmap_video_file, seg_video_file, findings

    
    #return heatmaps, seg_overlays, findings

# ---------------- GRADIO ----------------
with gr.Blocks(title="Endoscopy Video Classifier & Polyp Segmentation") as iface:
    gr.Markdown("# Endoscopy Video Classifier & Polyp Segmentation")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload Endoscopy Video")
        with gr.Column(scale=1):
            video_heatmap = gr.Video(label="Classification + GradCAM Overlay")
            video_seg = gr.Video(label="Polyp Segmentation Overlay (if polyp)")
            video_findings = gr.JSON(label="Frame-wise Key Findings")
    
    input_video.change(
        lambda v: process_video(v),
        inputs=[input_video],
        outputs=[video_heatmap, video_seg, video_findings]
    )

if __name__ == "__main__":
    iface.launch()
