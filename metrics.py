import torch
import torch.nn.functional as F

def compute_recall_at_k(text_emb, image_emb, ks=[1, 5, 10]):
    """
    Computes retrieval metrics (Recall@k) for text-image alignment.
    Assumes a one-to-one correspondence between text and image embeddings.
    """
    text_norm = F.normalize(text_emb, dim=1)
    image_norm = F.normalize(image_emb, dim=1)
    sim_matrix = torch.matmul(text_norm, image_norm.t())
    
    recalls = {}
    for k in ks:
        topk = sim_matrix.topk(k, dim=1).indices
        correct = 0
        for i in range(sim_matrix.size(0)):
            if i in topk[i]:
                correct += 1
        recalls[f"R@{k}"] = correct / sim_matrix.size(0)
    return recalls
