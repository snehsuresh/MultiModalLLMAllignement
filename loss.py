import torch
import torch.nn.functional as F
import torch.nn as nn

class DynamicAlignmentLoss(nn.Module):
    """
    Computes a contrastive loss between text and image embeddings,
    dynamically weighting the two directions based on a selected strategy.
    
    Weighting Strategies:
    - 'fixed': Fixed weighting (0.5-0.5).
    - 'variance': Weight computed based on the variance of embeddings.
    - 'entropy': Weight computed based on the entropy of embeddings.
    - 'cosine_spread': Weight computed based on intra-modal cosine spread.
    
    **Hypothesis:** Variance in the embeddings is a proxy for alignment uncertainty. 
    If one modality has higher variance, it indicates less confidence in its representation. 
    Dynamically adjusting the weighting (instead of fixed 50/50) can therefore stabilize and 
    accelerate the alignment between modalities, especially in low-resource settings.
    """
    def __init__(self, temperature=0.07, weighting_strategy='variance'):
        super(DynamicAlignmentLoss, self).__init__()
        self.temperature = temperature
        self.weighting_strategy = weighting_strategy
    
    def forward(self, text_emb, image_emb):
        # Normalize embeddings.
        text_norm = F.normalize(text_emb, dim=1)
        image_norm = F.normalize(image_emb, dim=1)
        # Cosine similarity matrix scaled by temperature.
        logits = torch.matmul(text_norm, image_norm.t()) / self.temperature
        batch_size = text_emb.size(0)
        labels = torch.arange(batch_size, device=text_emb.device)
        loss_text_to_image = F.cross_entropy(logits, labels)
        loss_image_to_text = F.cross_entropy(logits.t(), labels)
        base_loss = (loss_text_to_image + loss_image_to_text) / 2

        # Choose weighting strategy.
        if self.weighting_strategy == 'fixed':
            weight = 0.5
        elif self.weighting_strategy == 'variance':
            text_var = torch.var(text_emb, dim=0).mean()
            image_var = torch.var(image_emb, dim=0).mean()
            weight = text_var / (text_var + image_var + 1e-8)
        elif self.weighting_strategy == 'entropy':
            # Compute entropy-like measure for each modality.
            text_probs = F.softmax(text_emb, dim=1)
            image_probs = F.softmax(image_emb, dim=1)
            text_entropy = -torch.sum(text_probs * torch.log(text_probs + 1e-8), dim=1).mean()
            image_entropy = -torch.sum(image_probs * torch.log(image_probs + 1e-8), dim=1).mean()
            weight = image_entropy / (text_entropy + image_entropy + 1e-8)
        elif self.weighting_strategy == 'cosine_spread':
            # Calculate average intra-modal cosine similarity (excluding self-similarity).
            text_cosine = torch.matmul(text_norm, text_norm.t())
            image_cosine = torch.matmul(image_norm, image_norm.t())
            text_spread = (torch.sum(torch.abs(text_cosine - torch.eye(text_cosine.size(0), device=text_cosine.device))) /
                           (text_cosine.numel() - text_cosine.size(0)))
            image_spread = (torch.sum(torch.abs(image_cosine - torch.eye(image_cosine.size(0), device=image_cosine.device))) /
                            (image_cosine.numel() - image_cosine.size(0)))
            weight = text_spread / (text_spread + image_spread + 1e-8)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

        loss = weight * loss_text_to_image + (1 - weight) * loss_image_to_text + base_loss
        return loss
