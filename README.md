# **Variance-Aware Loss Scheduling For Multimodal Alignment In Low-Data Settings**

## **Overview**
This repository contains code, data, and results for our paper: "Variance-Aware Loss Scheduling For Multimodal Alignment In Low-Data Settings". We present a **multimodal alignment framework** that dynamically integrates **text and image embeddings** using both **custom-trained LSTM-CNN models** and **pretrained CLIP/BERT embeddings**. The goal is to **enhance text-image alignment** through **dynamic loss functions, contrastive learning, and visualization techniques**.

Key Features:
- **Switch betIen custom and pretrained embeddings** (LSTM-CNN vs. CLIP/BERT).
- **Contrastive loss for better embedding alignment.**
- **t-SNE visualization of embeddings with different perplexity settings.**
- **Supports scaling with larger datasets and extended training epochs.**

---

## **Installation**

Clone the repository:
```sh
git clone https://github.com/your-username/multimodal-alignment.git
cd multimodal-alignment
```

## **Dataset**
I use **Flickr8k** for text-image alignment. You can download it from:
- [Flickr8k Dataset](http://cs.stanford.edu/people/karpathy/deepimagesent/)
- Ensure that:
  - Images are stored in `data/flickr8k/Images`
  - Captions are in `data/flickr8k/captions.txt`

---

## **Usage**

### **1. Training the Model**
Train with either **custom embeddings** (LSTM-CNN) or **pretrained embeddings** (CLIP/BERT):

```sh
python main.py --use_pretrained True  # Uses CLIP/BERT
```
or
```sh
python main.py --use_pretrained False  # Uses LSTM-CNN
```

Adjust training parameters inside `main.py`:
- **Batch size**
- **Learning rate decay**
- **Dataset size**
- **Epochs** (default: `30`)

---

### **2. Visualizing Embeddings**
To generate **t-SNE projections** of text and image embeddings:

```sh
python visualize.py --perplexities 5 10 30 50
```
This will generate `tsne_perplexity_5.png`, `tsne_perplexity_10.png`, etc.

---

## **Results**

### **Training Performance**
| Model | Epochs | Loss | Cosine Similarity |
|--------|--------|------|------------------|
| LSTM-CNN | 10 | 1.68 | 0.58 |
| CLIP/BERT | 30 | **0.78** | **0.65** |

### **t-SNE Visualizations**
t-SNE plots showing text (blue) and image (red) embeddings for different perplexity values:

#### **Perplexity = 5**
![Perplexity 5](tsne_perplexity_5.png)

#### **Perplexity = 10**
![Perplexity 10](tsne_perplexity_10.png)

#### **Perplexity = 30 (Best)**
![Perplexity 30](tsne_perplexity_30.png)

#### **Perplexity = 50**
![Perplexity 50](tsne_perplexity_50.png)

---

## **Technical Details**

### **1. Custom Model Architecture**
When `use_pretrained=False`, I use:
- **LSTM for text processing**
- **ResNet-18 CNN for image encoding**
- **Fully Connected layers for multimodal fusion**

### **2. Pretrained Model Architecture**
When `use_pretrained=True`, I utilize:
- **CLIP's Vision Transformer (ViT-B/32) for image embeddings**
- **BERT (HuggingFace) for text embeddings**
- **Cosine similarity-based contrastive loss**

### **3. Loss Function**
Contrastive loss formulation:
```math
L = - \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(t_i, v_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(t_i, v_j) / \tau)}
```
where:
- **sim** is cosine similarity.
- **τ** is the temperature parameter.

---

## **Future Work**
- **Scale to COCO dataset** (larger image-text pairs).
- **Implement adversarial contrastive training** for robustness.
- **Experiment with different loss functions (e.g., NT-Xent).**

---

## **Citation**
If you use this code in your research, please cite:

```
@misc{multimodal2025,
  author = {Sneh Pillai},
  title = {Multimodal Alignment with Pretrained CLIP/BERT},
  year = {2025},
  url = {https://github.com/your-username/multimodal-alignment}
}
```

---

## **Contact**
For questions or collaborations, reach out to **Sneh Pillai** via GitHub Issues or email.

---
⭐ **Star this repo if you found it useful!**
