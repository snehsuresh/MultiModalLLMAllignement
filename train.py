import torch
import torch.nn.functional as F

def train(model, dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for batch_idx, (captions, lengths, images) in enumerate(dataloader):
            captions = captions.to(device)
            lengths = lengths.to(device)
            images = images.to(device)
            
            optimizer.zero_grad()
            text_emb, image_emb = model(captions, lengths, images)
            loss = criterion(text_emb, image_emb)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (batch_idx+1) % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"--- Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f} ---")
    return model

def verify_alignment(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        captions, lengths, images = next(iter(dataloader))
        captions = captions.to(device)
        lengths = lengths.to(device)
        images = images.to(device)
        text_emb, image_emb = model(captions, lengths, images)
        text_norm = F.normalize(text_emb, dim=1)
        image_norm = F.normalize(image_emb, dim=1)
        cosine_sim = (text_norm * image_norm).sum(dim=1)
        print("\nSample Cosine Similarity (Text-Image) for first 10 samples:")
        print(cosine_sim[:10].cpu().numpy())
        sim_matrix = torch.matmul(text_norm, image_norm.t())
        print("\nCosine Similarity Matrix (first 5 samples):")
        print(sim_matrix[:5, :5].cpu().numpy())
