import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("metrics/metrics.csv")

# Plot Loss Over Strategies
plt.figure(figsize=(8, 5))
plt.bar(df['loss_strategy'], df['final_loss'], color='skyblue')
plt.xlabel("Loss Strategy")
plt.ylabel("Final Loss (Lower = Better)")
plt.title("Final Loss After 30 Epochs")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("final_loss_comparison.png", dpi=300)
plt.show()

# Plot Retrieval Performance
plt.figure(figsize=(8, 5))
for k in ['R@1', 'R@5', 'R@10']:
    plt.plot(df['loss_strategy'], df[k], marker='o', label=k)

plt.xlabel("Loss Strategy")
plt.ylabel("Recall")
plt.title("Retrieval Performance")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("retrieval_performance.png", dpi=300)
plt.show()

# Plot Clean vs Noisy Loss
plt.figure(figsize=(8, 5))
plt.plot(df['loss_strategy'], df['clean_loss'], marker='o', label='Clean Loss')
plt.plot(df['loss_strategy'], df['noisy_loss'], marker='o', label='Noisy Loss')
plt.xlabel("Loss Strategy")
plt.ylabel("Loss")
plt.title("Stress Test (Clean vs Noisy)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("stress_test_loss.png", dpi=300)
plt.show()

# Plot Cosine Similarity Mean & Std
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Loss Strategy')
ax1.set_ylabel('Cosine Sim Mean', color=color)
ax1.plot(df['loss_strategy'], df['cosine_sim_mean'], marker='o', color=color, label='Cosine Sim Mean')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Cosine Sim Std', color=color)
ax2.plot(df['loss_strategy'], df['cosine_sim_std'], marker='o', color=color, linestyle='--', label='Cosine Sim Std')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle("Cosine Similarity Analysis")
fig.tight_layout()
plt.grid(True)
plt.xticks(rotation=45)
plt.savefig("cosine_similarity_analysis.png", dpi=300)
plt.show()
