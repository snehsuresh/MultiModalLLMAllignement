import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x, lengths):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # Pack the sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, _) = self.lstm(packed)
        # hidden: [1, batch_size, hidden_dim] → squeeze to [batch_size, hidden_dim]
        return hidden.squeeze(0)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ImageEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, H/2, W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, H/4, W/4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# [B, 128, H/8, W/8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, embed_dim)
    
    def forward(self, x):
        features = self.cnn(x)  # [B, 128, 1, 1]
        features = features.view(x.size(0), -1)  # [B, 128]
        embedding = self.fc(features)  # [B, embed_dim]
        return embedding

class MultiModalModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(MultiModalModel, self).__init__()
        self.text_encoder = TextEncoder(vocab_size, embed_dim, hidden_dim)
        self.image_encoder = ImageEncoder(embed_dim)
    
    def forward(self, captions, lengths, images):
        text_emb = self.text_encoder(captions, lengths)  # [B, hidden_dim]
        image_emb = self.image_encoder(images)           # [B, embed_dim]
        return text_emb, image_emb
