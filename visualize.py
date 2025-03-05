import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from dataset import FlickrDataset
from model import MultiModalModel

def load_data(img_dir, captions_file, transform, max_samples=100, use_pretrained=False):
    dataset = FlickrDataset(img_dir, captions_file, transform=transform, max_samples=max_samples, use_pretrained=use_pretrained)
    dataloader = DataLoader(dataset, batch_size=max_samples, shuffle=False)
    return dataset, dataloader

def visualize_embeddings(model, dataloader, device, use_pretrained, perplexities=[5, 10, 30, 50]):
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        if use_pretrained:
            captions, images = batch
            images = images.to(device)
            text_emb, image_emb = model(captions, images, lengths=None)
        else:
            captions, lengths, images = batch
            captions, lengths = captions.to(device), lengths.to(device)
            images = images.to(device)
            text_emb, image_emb = model(captions, images, lengths)
        
        # Normalize embeddings
        text_norm = text_emb.cpu().numpy()
        image_norm = image_emb.cpu().numpy()
        
        # Combine embeddings and create labels for plotting
        combined = np.concatenate([text_norm, image_norm], axis=0)
        labels = np.array([0]*len(text_norm) + [1]*len(image_norm))  # 0: text, 1: image

        # Iterate through different perplexity values
        for perplexity in perplexities:
            print(f"Generating t-SNE visualization with perplexity = {perplexity}...")
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced = tsne.fit_transform(combined)

            # Plot the embeddings
            plt.figure(figsize=(6, 4))
            plt.scatter(reduced[labels == 0, 0], reduced[labels == 0, 1], color='blue', label='Text Embeddings', alpha=0.6)
            plt.scatter(reduced[labels == 1, 0], reduced[labels == 1, 1], color='red', label='Image Embeddings', alpha=0.6)
            plt.legend()
            plt.title(f"t-SNE Visualization (Perplexity={perplexity})")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            save_path = f"embedding_visualization_p{perplexity}.png"
            plt.savefig(save_path, dpi=300)
            print(f"Saved t-SNE visualization as {save_path}")
            plt.close()

def main():
    use_pretrained = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 512 if use_pretrained else 256

    img_dir = "/scratch3/workspace/spillai_umassd_edu-pfolio/MultiModalAllignment/data/adityajn105/flickr8k/versions/1/Images"
    captions_file = "/scratch3/workspace/spillai_umassd_edu-pfolio/MultiModalAllignment/data/adityajn105/flickr8k/versions/1/captions.txt"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)) if use_pretrained else transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if use_pretrained else transforms.Normalize((0.5,), (0.5,))
    ])
    
    _, dataloader = load_data(img_dir, captions_file, transform, max_samples=100, use_pretrained=use_pretrained)
    
    model = MultiModalModel(embed_dim, use_pretrained=use_pretrained).to(device)
    model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
    print("Loaded trained model from multimodal_model.pth")
    
    visualize_embeddings(model, dataloader, device, use_pretrained)

if __name__ == "__main__":
    main()
