import torch
import torch.nn.functional as F

def collate_fn(batch, use_pretrained):
    """
    Custom collate function.
    Handles both pretrained (BERT/CLIP) and non-pretrained (LSTM/CNN) cases.
    """
    if use_pretrained:
        # If batch is already structured correctly, return it
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], list):
            return batch
        captions, images = zip(*batch)
        images = torch.stack(images)
        return list(captions), images
    else:
        # Handle non-pretrained: pad variable-length sequences
        if not isinstance(batch, list) or not (isinstance(batch[0], tuple) and len(batch[0]) == 2):
            raise ValueError(f"Unexpected batch structure: {batch}. Expected (caption, image).")
        
        captions, images = zip(*batch)
        lengths = [len(cap) for cap in captions]
        max_len = max(lengths)
        padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
        for i, cap in enumerate(captions):
            padded_captions[i, :len(cap)] = torch.tensor(cap, dtype=torch.long)
        images = torch.stack(images)
        return padded_captions, torch.tensor(lengths), images

def train(model, dataloader, optimizer, criterion, epochs, device, use_pretrained, scheduler):
    """
    Training loop with learning rate decay.
    """
    model.train()
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = collate_fn(batch, use_pretrained)
            except ValueError as e:
                print(f"Batch processing error: {e}")
                continue  # Skip problematic batch

            if use_pretrained:
                captions, images = batch
                images = images.to(device)
            else:
                captions, lengths, images = batch
                captions, lengths = captions.to(device), lengths.to(device)
                images = images.to(device)

            optimizer.zero_grad()
            text_emb, image_emb = model(captions, images, lengths if not use_pretrained else None)
            loss = criterion(text_emb, image_emb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx+1) % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"--- Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f} ---")

        # **Apply learning rate decay**
        scheduler.step(avg_loss)

    return model

def verify_alignment(model, dataloader, device, use_pretrained):
    """
    Verifies alignment by computing cosine similarities between text and image embeddings.
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch = collate_fn(batch, use_pretrained)
        
        if use_pretrained:
            captions, images = batch
            images = images.to(device)
        else:
            captions, lengths, images = batch
            captions, lengths = captions.to(device), lengths.to(device)
            images = images.to(device)

        text_emb, image_emb = model(captions, images, lengths if not use_pretrained else None)

        text_norm = F.normalize(text_emb, dim=1)
        image_norm = F.normalize(image_emb, dim=1)

        cosine_sim = (text_norm * image_norm).sum(dim=1)
        print("\nSample Cosine Similarity (Text-Image) for first 10 samples:")
        print(cosine_sim[:10].cpu().numpy())

        sim_matrix = torch.matmul(text_norm, image_norm.t())
        print("\nCosine Similarity Matrix (first 5 samples):")
        print(sim_matrix[:5, :5].cpu().numpy())
