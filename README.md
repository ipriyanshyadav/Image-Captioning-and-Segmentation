# Image Captioning and Segmentation

A deep learning project that generates captions for images and performs semantic segmentation using PyTorch.

## Features

- **Image Captioning**: Generate descriptive captions using CNN-RNN architecture
- **Image Segmentation**: Perform semantic segmentation using DeepLabV3
- **Web Interface**: Streamlit app for easy interaction

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web App
```bash
streamlit run app.py
```

### Python Scripts
```python
# Caption generation
from caption import generate_caption, encoder, decoder, word2idx, idx2word
caption = generate_caption("image.jpg", encoder, decoder, word2idx, idx2word)

# Segmentation
from segment import predict_and_visualize_cv2
predict_and_visualize_cv2("image.jpg")
```

## Models

- **Caption Model**: ResNet50 encoder + LSTM decoder
- **Segmentation Model**: DeepLabV3 with ResNet50 backbone

## Dataset

Place your datasets in the `Datasets/` directory following the existing structure.

### Dataset Links

- **Image Captioning Dataset**: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Image Segmentation Dataset**: [Supervisely Person Dataset](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset)

