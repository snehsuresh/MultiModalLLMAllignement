import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class FlickrDataset(Dataset):
    def __init__(self, img_dir, captions_file, transform=None, max_samples=None):
        """
        Args:
            img_dir (str): Directory with all the images.
            captions_file (str): Path to the file containing image names and captions.
            transform (callable, optional): Optional transform to be applied on an image.
            max_samples (int, optional): Use a subset of the data for demonstration.
        """
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.data = []  # List of tuples: (image filename, caption)
        with open(captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, caption = parts
                    self.data.append((img_name, caption))
        if max_samples is not None:
            self.data = self.data[:max_samples]
        # Build vocabulary from all captions.
        self.word2idx, self.idx2word = self.build_vocab([caption for _, caption in self.data])
        
    def build_vocab(self, captions):
        vocab = set()
        for caption in captions:
            vocab.update(caption.lower().split())
        # Add special tokens.
        vocab = ['<pad>', '<start>', '<end>', '<unk>'] + sorted(list(vocab))
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
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        caption_indices = self.tokenize_caption(caption)
        return caption_indices, image

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
