"""
FastAPI backend for Brain Tumor MRI classification + Grad-CAM localization.
Run:  uvicorn app:app --reload --port 8000
Needs in this folder: brain_tumor_mobilenetv2.keras + class_names.json
"""
import base64, io, json, os, traceback
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_mobilenetv2.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (224, 224)
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DISPLAY_MAX = 600             # max side (px) of the returned heatmap image

if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
    raise RuntimeError("Model files missing. Train with model/train.py and copy "
                       "brain_tumor_mobilenetv2.keras and class_names.json into backend/.")

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor Detected",
}

# ----------------------------------------------------------------------
# Grad-CAM setup: split the model into  [pre layers] -> MobileNetV2 -> [head layers]
# so we can take gradients of the class score w.r.t. MobileNet's last feature map.
# ----------------------------------------------------------------------
_layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
_nested = [i for i, l in enumerate(_layers) if isinstance(l, tf.keras.Model)]
_base_idx = max(_nested, key=lambda i: len(_layers[i].layers))  # biggest nested model = MobileNetV2
BASE = _layers[_base_idx]
PRE_LAYERS = _layers[:_base_idx]
HEAD_LAYERS = _layers[_base_idx + 1:]


def _run(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def compute_gradcam(batch, class_idx):
    """Returns a 7x7 heatmap in [0, 1]."""
    x = tf.convert_to_tensor(batch)
    for l in PRE_LAYERS:
        x = _run(l, x)
    with tf.GradientTape() as tape:
        conv = _run(BASE, x)          # (1, 7, 7, 1280) last conv feature map
        tape.watch(conv)
        y = conv
        for l in HEAD_LAYERS:
            y = _run(l, y)
        score = y[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv, axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def jet_colormap(v):
    """v: float array in [0,1] -> RGB float array (H,W,3) in [0,1] (blue -> green -> red)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def region_name(cx, cy):
    h = "left" if cx < 0.38 else "right" if cx > 0.62 else "center"
    v = "upper" if cy < 0.38 else "lower" if cy > 0.62 else "middle"
    if h == "center" and v == "middle":
        return "center of the scan"
    if h == "center":
        return f"{v} part of the scan"
    if v == "middle":
        return f"{h} side of the scan"
    return f"{v}-{h} region of the scan"


def build_overlay(img_rgb, cam, draw_box):
    """Blend the colored heatmap onto the MRI. Returns (data_url, location_text_or_None)."""
    w, h = img_rgb.size
    scale = min(1.0, DISPLAY_MAX / max(w, h))
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    base = img_rgb.resize(size, Image.BICUBIC)

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BICUBIC)
    cam_big = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = jet_colormap(cam_big)
    blended = 0.55 * (np.asarray(base, dtype=np.float32) / 255.0) + 0.45 * heat
    out = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))

    location = None
    if draw_box:
        ys, xs = np.where(cam_big >= 0.7)
        if len(xs) > 0:
            draw = ImageDraw.Draw(out)
            draw.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=(255, 255, 255), width=3)
            location = region_name(xs.mean() / size[0], ys.mean() / size[1])

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), location


# ----------------------------------------------------------------------
app = FastAPI(title="Brain Tumor Detection API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"message": "Brain Tumor Detection API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file (JPG/PNG).")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")
    try:
        img_full = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the image file.")

    img = img_full.resize(IMG_SIZE)
    arr = np.expand_dims(np.asarray(img, dtype=np.float32), 0)  # raw 0-255; scaling is inside the model
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    has_tumor = CLASS_NAMES[idx] != "notumor"

    gradcam, location = None, None
    try:
        cam = compute_gradcam(arr, idx)
        gradcam, location = build_overlay(img_full, cam, draw_box=has_tumor)
    except Exception:
        traceback.print_exc()   # prediction still works even if Grad-CAM fails

    return {
        "prediction": CLASS_NAMES[idx],
        "label": LABELS.get(CLASS_NAMES[idx], CLASS_NAMES[idx]),
        "confidence": round(float(probs[idx]) * 100, 2),
        "has_tumor": has_tumor,
        "probabilities": {
            LABELS.get(c, c): round(float(p) * 100, 2) for c, p in zip(CLASS_NAMES, probs)
        },
        "gradcam": gradcam,
        "location": location,
    }
