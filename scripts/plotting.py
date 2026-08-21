import torch
import matplotlib.pyplot as plt

# 1. Load data and orthogonalized directions directly from the root-level data folder
centered_data = torch.load("data/emotion_centered_vectors.pt", weights_only=True)
va_dirs = torch.load("data/emotion_va_directions_orthogonal.pt", weights_only=True)

# 2. Extract components for layer 14 spot-check
layer_idx = 14
V_centered = centered_data["V_centered_per_layer"][layer_idx]  # [27, 2048]
emotions = centered_data["emotions"]  # List of 27 emotion names

w_V = va_dirs["w_V_per_layer"][layer_idx]  # [2048]
w_A = va_dirs["w_A_per_layer"][layer_idx]  # [2048]

# 3. Project 27 mean-centered emotion vectors onto final 2D plane
valence_proj = torch.matmul(V_centered, w_V).cpu().numpy()
arousal_proj = torch.matmul(V_centered, w_A).cpu().numpy()

# 4. Plotting Russell's Circumplex space
plt.figure(figsize=(10, 10))
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)

plt.scatter(valence_proj, arousal_proj, color='royalblue', alpha=0.7)

for i, emotion in enumerate(emotions):
    plt.annotate(
        emotion, 
        (valence_proj[i], arousal_proj[i]),
        fontsize=9,
        alpha=0.85,
        ha='right', 
        va='bottom'
    )

plt.title(f"Russell's Circumplex Model (Layer {layer_idx})", fontsize=14)
plt.xlabel("Valence Projection (w_V)", fontsize=12)
plt.ylabel("Arousal Projection (w_A)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

# Save into the outputs folder
plt.savefig("outputs/circumplex_validation_layer14.png", dpi=300, bbox_inches='tight')
plt.show()

print("Circumplex projection complete! Plot saved to outputs/circumplex_validation_layer14.png")