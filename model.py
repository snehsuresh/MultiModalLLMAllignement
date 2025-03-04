import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import open_clip  # For CLIP
import torchvision.models as models

class TextEncoder(nn.Module):
    def __init__(self, embed_dim, use_pretrained=False, pretrained_model="bert-base-uncased", vocab_size=10000, hidden_dim=256):
        super(TextEncoder, self).__init__()
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # Use BERT for text embeddings
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.bert = BertModel.from_pretrained(pretrained_model)
            self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)
        else:
            # Custom LSTM-based text encoder
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, embed_dim)

        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, captions, lengths=None):
        if self.use_pretrained:
            # Tokenize captions and get BERT embeddings
            inputs = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)
            with torch.no_grad():
                outputs = self.bert(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token
            features = self.fc(cls_embedding)
        else:
            # Process captions with LSTM
            embedded = self.embedding(captions)
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
            features = self.fc(hidden.squeeze(0))

        return self.bn(features)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, use_pretrained=False, clip_model="ViT-B/32"):
        super(ImageEncoder, self).__init__()
        self.use_pretrained = use_pretrained

        if use_pretrained:
            # Use CLIP for image embeddings
            self.clip_model, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained="openai")
            self.clip_model = self.clip_model.visual
            self.fc = nn.Linear(self.clip_model.output_dim, embed_dim)
        else:
            # Custom CNN-based image encoder
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(128, embed_dim)

        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, images):
        if self.use_pretrained:
            with torch.no_grad():
                features = self.clip_model(images)
        else:
            features = self.cnn(images).view(images.size(0), -1)

        features = self.fc(features)
        return self.bn(features)

class MultiModalModel(nn.Module):
    def __init__(self, embed_dim, vocab_size=10000, hidden_dim=256, use_pretrained=False):
        super(MultiModalModel, self).__init__()
        self.text_encoder = TextEncoder(embed_dim, use_pretrained, vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.image_encoder = ImageEncoder(embed_dim, use_pretrained)

    def forward(self, captions, images, lengths=None):
        text_emb = self.text_encoder(captions, lengths)
        image_emb = self.image_encoder(images)
        return text_emb, image_emb
