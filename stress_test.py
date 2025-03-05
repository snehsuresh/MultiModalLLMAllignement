import torch
from train import collate_fn  # reuse your collate function

def stress_test(model, dataloader, criterion, device, use_pretrained, noise_level=0.1):
    """
    Performs a stress test by injecting noise into the inputs (images or captions) 
    and evaluating how the loss behaves compared to clean inputs.
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch = collate_fn(batch, use_pretrained)
        
        if use_pretrained:
            captions, images = batch
            images = images.to(device)
            # Inject Gaussian noise into images.
            noisy_images = images + noise_level * torch.randn_like(images)
            noisy_images = noisy_images.to(device)
            text_emb_clean, image_emb_clean = model(captions, images, None)
            text_emb_noisy, image_emb_noisy = model(captions, noisy_images, None)
        else:
            captions, lengths, images = batch
            captions = captions.to(device)
            lengths = lengths.to(device)
            images = images.to(device)
            # Inject noise into captions by randomly replacing tokens with <unk> (assuming index 3 is <unk>).
            noisy_captions = []
            for cap in captions:
                noisy_cap = cap.clone()
                mask = (torch.rand_like(noisy_cap.float()) < noise_level)
                noisy_cap[mask] = 3  # Replace with <unk>
                noisy_captions.append(noisy_cap)
            noisy_captions = torch.stack(noisy_captions).to(device)
            text_emb_clean, image_emb_clean = model(captions, images, lengths)
            text_emb_noisy, image_emb_noisy = model(noisy_captions, images, lengths)
        
        loss_clean = criterion(text_emb_clean, image_emb_clean)
        loss_noisy = criterion(text_emb_noisy, image_emb_noisy)
        
        print("Stress Test Results:")
        print(f"Clean Loss: {loss_clean.item():.4f}")
        print(f"Noisy Loss: {loss_noisy.item():.4f}")
        return loss_clean.item(), loss_noisy.item()
