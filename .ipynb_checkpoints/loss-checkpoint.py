import torch
import torch.nn.functional as F
import torch.nn as nn

class DynamicAlignmentLoss(nn.Module):
    """
    Computes a contrastive loss between text and image embeddings,
    dynamically weighting the two directions based on their embedding variance.
    """
    def __init__(self, temperature=0.07):
        super(DynamicAlignmentLoss, self).__init__()
        self.temperature = temperature
    
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
        
        # Dynamic weighting factor based on variance.
        text_var = torch.var(text_emb, dim=0).mean()
        image_var = torch.var(image_emb, dim=0).mean()
        dynamic_weight = text_var / (text_var + image_var + 1e-8)
        
        loss = dynamic_weight * loss_text_to_image + (1 - dynamic_weight) * loss_image_to_text + base_loss
        return loss
