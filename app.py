import streamlit as st
import os
import sys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Add project root to Python path for module imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import your modules
from caption import generate_caption, encoder, decoder, word2idx, idx2word, image_transform, device
from segment import predict_and_visualize_cv2, model as seg_model

st.title("Image Captioning and Segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to a temp file for OpenCV compatibility
    temp_path = "temp_uploaded_image.png"
    image.save(temp_path)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(
                    temp_path, encoder, decoder, word2idx, idx2word
                )
                st.success("Caption:")
                st.write(caption)

    with col2:
        if st.button("Generate Segmentation"):
            with st.spinner("Generating segmentation..."):
                # Use matplotlib to get the output as an image
                fig, ax = plt.subplots(figsize=(12, 5))
                # Call the segmentation function but redirect plt to this fig
                # We'll use the code from segment.py but adapt it for Streamlit
                import cv2

                image_bgr = cv2.imread(temp_path)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                transform = image_transform if "image_transform" in globals() else None
                if transform is None:
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                    ])
                input_tensor = transform(image_pil).unsqueeze(0).to(device)
                seg_model.eval()
                with torch.no_grad():
                    output = seg_model(input_tensor)['out']
                    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                pred_mask_resized = cv2.resize(pred_mask, (image_rgb.shape[1], image_rgb.shape[0]))
                binary_mask = (pred_mask_resized > 0.5).astype(np.uint8)
                overlay = image_rgb.copy()
                overlay[binary_mask == 1] = [0, 255, 0]
                blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
                # Show only the segmentation overlay
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.imshow(blended)
                ax.set_title("Prediction Overlay")
                ax.axis("off")
                st.pyplot(fig)

    # Remove temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)