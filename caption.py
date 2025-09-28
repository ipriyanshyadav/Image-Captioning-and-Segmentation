# DATASET -> https://www.kaggle.com/datasets/adityajn105/flickr8k


# Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from collections import Counter
from PIL import Image
import pickle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')


# Use GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


#Loading dataset
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, '..', 'dataset', 'Images')

df = pd.read_csv(os.path.join(project_root, '..', 'dataset', 'captions.txt'))
df.columns = ['image', 'caption']

def preprocess_caption(caption):
    return "startseq " + caption.lower().strip() + " endseq"

df['caption'] = df['caption'].apply(preprocess_caption)


# Vocabulary
def build_vocab(captions, threshold=5):
    counter = Counter()
    for cap in captions:
        counter.update(cap.split())

    vocab = {word for word in counter if counter[word] >= threshold}
    word2idx = {word: idx+1 for idx, word in enumerate(vocab)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = len(word2idx) + 1

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

# Check if vocab files exist
if os.path.exists("word2idx.pkl") and os.path.exists("idx2word.pkl"):
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("idx2word.pkl", "rb") as f:
        idx2word = pickle.load(f)
else:
    word2idx, idx2word = build_vocab(df['caption'])
    # Save vocab for future use
    with open("word2idx.pkl", "wb") as f:
        pickle.dump(word2idx, f)
    with open("idx2word.pkl", "wb") as f:
        pickle.dump(idx2word, f)

vocab_size = len(word2idx)


# Transform and Dataset Class
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FlickrDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.images = list(df['image'].unique())
        self.captions_dict = df.groupby('image')['caption'].apply(list).to_dict()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        captions = self.captions_dict[image_name]
        return image, captions[0]  # using first caption


# Data Loader
image_dir = data_dir
train_df, valid_df = train_test_split(df, test_size=0.1, shuffle=True)

train_dataset = FlickrDataset(train_df, image_dir, image_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)


# Encoder Model
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)  # Always [batch, features]
        features = self.linear(features)
        features = self.bn(features)
        return features


# Decoder Model
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs


# Loss and Optimizer
encoder = EncoderCNN(embed_size=256).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)


# Training Loop
num_epochs = 20

encoder_weights = "caption-encoder.pth"
decoder_weights = "caption-decoder.pth"

def caption_to_indices(caption, word2idx):
    tokens = caption.split()
    return [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

# Only train if weights do not exist
if not (os.path.exists(encoder_weights) and os.path.exists(decoder_weights)):
    print("Training model...")
    # --- Training loop ---
    for epoch in range(num_epochs):
        for i, (images, captions) in enumerate(train_loader):
            # ...existing training code...
            caption_indices = [torch.tensor(caption_to_indices(c, word2idx)) for c in captions]
            caption_indices = pad_sequence(caption_indices, batch_first=True, padding_value=word2idx['<PAD>'])
            captions_input = caption_indices[:, :-1].to(device)
            captions_target = caption_indices[:, 1:].to(device)
            images = images.to(device)
            features = encoder(images)
            outputs = decoder(features, captions_input)
            outputs = outputs[:, 1:, :]
            loss = criterion(outputs.reshape(-1, vocab_size), captions_target.reshape(-1))
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}], Loss: {loss.item():.4f}")
    # --- Save weights ---
    torch.save(encoder.state_dict(), encoder_weights)
    torch.save(decoder.state_dict(), decoder_weights)
    print("Training complete and weights saved.")
else: 
    print("Weights found, skipping training.")


# Loading  weights before inference
encoder.load_state_dict(torch.load(encoder_weights, map_location=device))
decoder.load_state_dict(torch.load(decoder_weights, map_location=device))
encoder.eval()
decoder.eval()


# generate captions
def generate_caption(image_path, encoder, decoder, word2idx, idx2word, max_len=20):

    image = Image.open(image_path).convert('RGB')
    image = image_transform(image)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        feature = encoder(image)
        caption = []
        word = torch.tensor([word2idx['startseq']]).to(device)
        hidden = None

        for _ in range(max_len):
            if hidden is None:
                # First input: concat feature and embedding
                inputs = torch.cat((feature.unsqueeze(1), decoder.embed(word).unsqueeze(1)), dim=1)
                hiddens, hidden = decoder.lstm(inputs)
                output = decoder.linear(hiddens[:, -1, :])
            else:
                # Next input: only embedding of last word
                embedded = decoder.embed(word).unsqueeze(1)
                hiddens, hidden = decoder.lstm(embedded, hidden)
                output = decoder.linear(hiddens.squeeze(1))

            predicted = output.argmax(-1)
            word_idx = predicted.item()
            word_str = idx2word.get(word_idx, '<UNK>')
            if word_str == 'endseq':
                break
            caption.append(word_str)
            word = predicted

    # Remove 'startseq' if present at the beginning
    if caption and caption[0] == 'startseq':
        caption = caption[1:]
    # Remove 'endseq' if present at the end
    if caption and caption[-1] == 'endseq':
        caption = caption[:-1]
    return ' '.join(caption)

def result(image_path, encoder, decoder, word2idx, idx2word):
    caption = generate_caption(image_path, encoder, decoder, word2idx, idx2word)
    print("Generated Caption:", caption)
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Image")
    plt.show()

def input_image(image_loc):
    b= image_loc
    a= result(b, encoder, decoder, word2idx, idx2word)