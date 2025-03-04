import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FlickrDataset
from model import MultiModalModel
from loss import DynamicAlignmentLoss
from train import train, verify_alignment, collate_fn

def main():
    use_pretrained = True  # Set this to False to use LSTM/CNN instead of BERT/CLIP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 512 if use_pretrained else 256
    batch_size = 32 if use_pretrained else 16  # Adjust batch size for CLIP/BERT memory usage
    epochs = 30  # Increased to 30 epochs for better alignment
    initial_lr = 1e-4 if use_pretrained else 1e-3

    img_dir = "/scratch3/workspace/spillai_umassd_edu-pfolio/MultiModalAllignment/data/adityajn105/flickr8k/versions/1/Images"
    captions_file = "/scratch3/workspace/spillai_umassd_edu-pfolio/MultiModalAllignment/data/adityajn105/flickr8k/versions/1/captions.txt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = FlickrDataset(img_dir, captions_file, transform=transform, max_samples=6000, use_pretrained=use_pretrained)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, use_pretrained)
    )

    model = MultiModalModel(embed_dim, use_pretrained=use_pretrained).to(device)
    criterion = DynamicAlignmentLoss(temperature=0.07)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # **Learning Rate Scheduler**
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    print(f"Training on device: {device} | Using Pretrained Models: {use_pretrained}")
    model = train(model, dataloader, optimizer, criterion, epochs, device, use_pretrained, scheduler)

    verify_alignment(model, dataloader, device, use_pretrained)  # Pass use_pretrained flag
    torch.save(model.state_dict(), "multimodal_model.pth")
    print("Model saved as multimodal_model.pth")

if __name__ == "__main__":
    main()
