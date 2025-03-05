import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FlickrDataset
from model import MultiModalModel
from loss import DynamicAlignmentLoss
from train import train, verify_alignment, collate_fn
from metrics import compute_recall_at_k
from stress_test import stress_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pretrained", action="store_true", help="Use BERT/CLIP for pretrained embeddings")
    parser.add_argument("--loss_strategy", type=str, default="variance", 
                        choices=["fixed", "variance", "entropy", "cosine_spread"], 
                        help="Loss weighting strategy")
    args = parser.parse_args()

    use_pretrained = args.use_pretrained
    weighting_strategy = args.loss_strategy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 512 if use_pretrained else 256
    batch_size = 32 if use_pretrained else 16
    epochs = 30
    initial_lr = 1e-4 if use_pretrained else 1e-3

    img_dir = "/scratch3/workspace/spillai_umassd_edu-pfolio/MultiModalAllignment/data/adityajn105/flickr8k/versions/1/Images"  # adjust to your paths
    captions_file = "/scratch3/workspace/spillai_umassd_edu-pfolio/MultiModalAllignment/data/adityajn105/flickr8k/versions/1/captions.txt"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = FlickrDataset(img_dir, captions_file, transform=transform, max_samples=2000, use_pretrained=use_pretrained)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, use_pretrained)
    )

    model = MultiModalModel(embed_dim, use_pretrained=use_pretrained).to(device)
    criterion = DynamicAlignmentLoss(temperature=0.07, weighting_strategy=weighting_strategy)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    print(f"Training on device: {device} | Using Pretrained: {use_pretrained} | Loss Strategy: {weighting_strategy}")
    model = train(model, dataloader, optimizer, criterion, epochs, device, use_pretrained, scheduler)

    print("\nVerifying Alignment:")
    verify_alignment(model, dataloader, device, use_pretrained)
    
    # Compute retrieval metrics.
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch = collate_fn(batch, use_pretrained)
        if use_pretrained:
            captions, images = batch
            images = images.to(device)
            text_emb, image_emb = model(captions, images, None)
        else:
            captions, lengths, images = batch
            captions, lengths = captions.to(device), lengths.to(device)
            images = images.to(device)
            text_emb, image_emb = model(captions, images, lengths)
        recalls = compute_recall_at_k(text_emb, image_emb, ks=[1, 5, 10])
        print("Retrieval Metrics:", recalls)
    
    # Run the stress test.
    print("\nRunning Stress Test (with injected noise):")
    stress_test(model, dataloader, criterion, device, use_pretrained, noise_level=0.1)
    
    torch.save(model.state_dict(), "multimodal_model.pth")
    print("Model saved as multimodal_model.pth")

if __name__ == "__main__":
    main()
