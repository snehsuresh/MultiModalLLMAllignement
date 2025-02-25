import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FlickrDataset, collate_fn
from model import MultiModalModel
from loss import DynamicAlignmentLoss
from train import train, verify_alignment

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 256
    hidden_dim = 256
    batch_size = 16
    epochs = 5  # For demo purposes; increase for more training.
    learning_rate = 1e-3

    # Data paths (update these paths to match your data)
    img_dir = "data/images"
    captions_file = "data/captions.txt"
    
    # Image transformation: resize images for consistency.
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Initialize dataset and dataloader.
    dataset = FlickrDataset(img_dir, captions_file, transform=transform, max_samples=200)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Get vocabulary size from dataset.
    vocab_size = len(dataset.word2idx)
    print("Vocabulary size:", vocab_size)
    
    # Initialize model, loss function, and optimizer.
    model = MultiModalModel(vocab_size, embed_dim, hidden_dim).to(device)
    criterion = DynamicAlignmentLoss(temperature=0.07)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on device: {device}")
    
    # Train the model.
    model = train(model, dataloader, optimizer, criterion, epochs, device)
    
    # Verify alignment between modalities.
    verify_alignment(model, dataloader, device)
    
    # Save the trained model.
    torch.save(model.state_dict(), "multimodal_model.pth")
    print("Model saved as multimodal_model.pth")

if __name__ == "__main__":
    main()
