import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class FlickrDataset(Dataset):
    def __init__(self, img_dir, captions_file, transform=None, max_samples=None, use_pretrained=False):
        """
        Args:
            img_dir (str): Directory containing images.
            captions_file (str): Path to the CSV file containing image names and captions.
            transform (callable, optional): Image transformations.
            max_samples (int, optional): Limit the dataset size for faster training/testing.
            use_pretrained (bool): If True, returns raw text captions (for BERT). If False, returns tokenized indices (for LSTM).
        """
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.use_pretrained = use_pretrained
        self.data = []

        with open(captions_file, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                if len(row) != 2:
                    continue
                img_name, caption = row
                img_path = os.path.join(self.img_dir, img_name.strip())

                if not os.path.exists(img_path):
                    continue

                self.data.append((img_name.strip(), caption.strip()))

        if max_samples:
            self.data = self.data[:max_samples]

        if not self.use_pretrained:
            # Build vocabulary only if using LSTM (not needed for BERT)
            self.word2idx, self.idx2word = self.build_vocab([caption for _, caption in self.data])

    def build_vocab(self, captions):
        vocab = set()
        for caption in captions:
            vocab.update(caption.lower().split())

        # Add special tokens
        vocab = ['<pad>', '<start>', '<end>', '<unk>'] + sorted(vocab)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word

    def tokenize_caption(self, caption):
        tokens = caption.lower().split()
        tokens = ['<start>'] + tokens + ['<end>']
        indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in tokens]
        return indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Open and transform the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.use_pretrained:
            return caption, image  # Return raw caption as string for BERT
        else:
            return self.tokenize_caption(caption), image  # Return tokenized caption for LSTM


def collate_fn(batch):
    """Pads variable-length caption sequences and stacks images."""
    captions, images = zip(*batch)
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = torch.tensor(cap, dtype=torch.long)
    
    images = torch.stack(images)

    return padded_captions, torch.tensor(lengths), images
