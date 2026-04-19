import sys
import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ── paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CAP_DIR  = os.path.join(ROOT_DIR, "Image Captioning")
SEG_DIR  = os.path.join(ROOT_DIR, "Image Segmentation")
sys.path.insert(0, CAP_DIR)

from data_loader import FlickrDataset
from generate_caption import EncoderDecoder, load_model, generate_caption

# ── device ───────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

device = get_device()

# ── sample images ─────────────────────────────────────────────────────────────
SAMPLE_DIR  = os.path.join(ROOT_DIR, "sample_images")
CAP_SAMPLE  = os.path.join(SAMPLE_DIR, "captioning")
SEG_SAMPLE  = os.path.join(SAMPLE_DIR, "segmentation")

_IMG_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(folder):
    return sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in _IMG_EXTS)

# ── loaders (cached) ──────────────────────────────────────────────────────────
@st.cache_resource
def load_caption_model():
    cap_transform = T.Compose([
        T.Resize(226), T.CenterCrop(224), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = FlickrDataset(
        root_dir=os.path.join(CAP_DIR, "archive/Images"),
        caption_file=os.path.join(CAP_DIR, "archive/captions.txt"),
        transform=cap_transform,
    )
    model = load_model(os.path.join(CAP_DIR, "attention_model_state.pth"), dataset.vocab)
    return model, dataset.vocab, cap_transform

@st.cache_resource
def load_seg_model():
    model = torch.load(
        os.path.join(SEG_DIR, "Unet-Mobilenet_v2_mIoU-0.227.pt"),
        map_location=device,
        weights_only=False
    )
    model.to(device)
    model.eval()
    return model

# ── helpers ───────────────────────────────────────────────────────────────────
def run_caption(pil_img):
    model, vocab, transform = load_caption_model()
    img_tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encoder(img_tensor)
        caps, _ = model.decoder.generate_caption(features, vocab=vocab)
    words = [w for w in caps if w not in ("<SOS>", "<EOS>", "<PAD>")]
    return " ".join(words)

def run_segmentation(pil_img):
    seg_model = load_seg_model()
    img = np.array(pil_img.convert("RGB"))
    img_resized = np.array(pil_img.convert("RGB").resize((1152, 768), Image.NEAREST))
    t = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    inp = t(Image.fromarray(img_resized)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = seg_model(inp)
        mask = torch.argmax(F.softmax(out, dim=1), dim=1).cpu().squeeze(0).numpy()
    # Normalize mask to 0-255 for display
    mask_display = (mask / mask.max() * 255).astype(np.uint8) if mask.max() > 0 else mask.astype(np.uint8)
    return img, mask_display

def image_grid(image_paths, key_prefix, cols_per_row=7):
    """Render thumbnail images in rows; return index of clicked one."""
    selected = None
    for row_start in range(0, len(image_paths), cols_per_row):
        batch = image_paths[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, (col, path) in enumerate(zip(cols, batch)):
            img = Image.open(path)
            col.image(img, use_container_width=True)
            if col.button("Select", key=f"{key_prefix}_{row_start + i}"):
                selected = row_start + i
    return selected

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Image Captioning and Segmentation", layout="wide")
st.title("Image Captioning and Segmentation")

tab_cap, tab_seg = st.tabs(["🖼️ Image Captioning", "🎨 Image Segmentation"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Image Captioning
# ════════════════════════════════════════════════════════════════════════════
with tab_cap:
    st.subheader("Image Captioning with Attention")

    cap_img_paths = [os.path.join(CAP_SAMPLE, f) for f in list_images(CAP_SAMPLE)]

    # Upload
    uploaded = st.file_uploader("Upload your own image", type=["jpg","jpeg","png"], key="cap_upload")

    cap_selected = image_grid(cap_img_paths, "cap")

    # Resolve active image
    cap_pil = None
    if uploaded:
        cap_pil = Image.open(io.BytesIO(uploaded.read()))
    elif cap_selected is not None:
        cap_pil = Image.open(cap_img_paths[cap_selected])
        st.session_state["cap_pil"] = cap_pil
    elif "cap_pil" in st.session_state:
        cap_pil = st.session_state["cap_pil"]

    if cap_pil:
        col1, col2 = st.columns([1, 2])
        col1.image(cap_pil, caption="Selected Image", use_container_width=True)
        with col2:
            if st.button("Generate Caption", type="primary"):
                with st.spinner("Generating caption…"):
                    caption = run_caption(cap_pil)
                st.success(f"**Caption:** {caption}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Image Segmentation
# ════════════════════════════════════════════════════════════════════════════
with tab_seg:
    st.subheader("Semantic Segmentation (Drone Dataset)")

    seg_img_paths = [os.path.join(SEG_SAMPLE, f) for f in list_images(SEG_SAMPLE)]

    # Upload
    seg_uploaded = st.file_uploader("Upload your own image", type=["jpg","jpeg","png"], key="seg_upload")

    seg_selected = image_grid(seg_img_paths, "seg")

    # Resolve active image
    seg_pil = None
    if seg_uploaded:
        seg_pil = Image.open(io.BytesIO(seg_uploaded.read()))
    elif seg_selected is not None:
        seg_pil = Image.open(seg_img_paths[seg_selected])
        st.session_state["seg_pil"] = seg_pil
    elif "seg_pil" in st.session_state:
        seg_pil = st.session_state["seg_pil"]

    if seg_pil:
        col1, col2, col3 = st.columns(3)
        col1.image(seg_pil, caption="Original Image", use_container_width=True)
        if col2.button("Run Segmentation", type="primary"):
            with st.spinner("Segmenting…"):
                original, mask = run_segmentation(seg_pil)
            col2.image(original, caption="Preprocessed", use_container_width=True)
            col3.image(mask, caption="Segmentation Mask", use_container_width=True, clamp=True)
