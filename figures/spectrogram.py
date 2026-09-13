import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate the Time Series Signal ---
Fs = 1000  # Sampling frequency
T = 5  # 5 seconds of data
t = np.linspace(0, T, Fs * T)

# Start with some base background noise (the dark blue gaps in your image)
signal = np.random.normal(0, 0.5, len(t))

# Add several constant frequencies to create those horizontal "bands"
# We'll give them different amplitudes to make some bands brighter green than others
harmonics = [
    {"freq": 120, "amp": 1.5},
    {"freq": 250, "amp": 0.8},
    {"freq": 380, "amp": 2.0},
    {"freq": 410, "amp": 1.2},
    {"freq": 600, "amp": 0.5},
    {"freq": 750, "amp": 1.8},
]

for h in harmonics:
    # Adding a tiny bit of random wobble to the amplitude makes it look more natural
    wobble = np.random.uniform(0.9, 1.1)
    signal += h["amp"] * wobble * np.sin(2 * np.pi * h["freq"] * t)

# --- 2. Setup the Square Figure ---
fig, ax = plt.subplots(figsize=(8, 8))

# --- 3. Generate the Blue-to-Green Spectrogram ---
NFFT = 256
noverlap = 128

# 'viridis' handles that exact dark-blue to bright-green transition beautifully
ax.specgram(signal, NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap="viridis")

# --- 4. Strip away all axes, ticks, and borders ---
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Remove all padding so it's a pure square of data
plt.tight_layout(pad=0)

# --- 5. Save as SVG ---
plt.savefig(
    "blue_green_bands_spectrogram.svg", format="svg", bbox_inches="tight", pad_inches=0
)

print("Spectrogram successfully saved as 'blue_green_bands_spectrogram.svg'!")
