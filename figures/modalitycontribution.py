import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    "Category": [
        "Anomaly Detection",
        "Causality Analysis",
        "Noise Understanding",
        "Pattern Recognition",
        "Similarity Analysis",
    ],
    "Vision": [21.86111111, 14.81555556, 40.13404762, 30.01160221, 30.07441667],
    "Math Tools": [25.35888889, 51.85444444, 17.00595238, 27.16839779, 20.75558333],
    "Total Accuracy": [47.22, 66.67, 57.14, 57.18, 50.83],
}

df = pd.DataFrame(data)
df = df.sort_values(by="Total Accuracy", ascending=True).reset_index(drop=True)

# Set up clean aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)

# Create layout figure
fig, ax = plt.subplots(figsize=(10, 6))

# Horizontal stacked bar chart matching proportions precisely
bars_math = ax.barh(
    df["Category"], df["Math Tools"], color="#1f77b4", edgecolor="none", label="Stats"
)
bars_vision = ax.barh(
    df["Category"],
    df["Vision"],
    left=df["Math Tools"],
    color="#ff7f0e",
    edgecolor="none",
    label="Vision",
)

# Annotating percentages directly inside the horizontal segments
for i in range(len(df)):
    math_val = df["Math Tools"][i]
    vision_val = df["Vision"][i]
    total_val = df["Total Accuracy"][i]

    # Text for Math Tools (Stats)
    if math_val > 5:
        ax.text(
            math_val / 2,
            i,
            f"{math_val:.1f}%",
            va="center",
            ha="center",
            color="white",
            fontweight="bold",
        )
    # Text for Vision
    if vision_val > 5:
        ax.text(
            math_val + (vision_val / 2),
            i,
            f"{vision_val:.1f}%",
            va="center",
            ha="center",
            color="white",
            fontweight="bold",
        )

    # Text for total final value
    ax.text(
        total_val + 1,
        i,
        f"Total: {total_val:.1f}%",
        va="center",
        ha="left",
        color="#2c3e50",
        fontweight="bold",
    )

# Applying user adjustments: Title with colon after Success, label cleanups
ax.set_xlabel("Accuracy (%)", labelpad=15)
ax.set_ylabel("Task Category", labelpad=15)
ax.set_title(
    "Modality Contribution to Framework Success",
    pad=20,
    fontweight="bold",
    color="#1a252f",
)
ax.set_xlim(0, 75)

# Minimalist design cleanup
sns.despine(left=True, bottom=True)

# Moving legend to lower right corner with shortened names
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()

# Save output to SVG only
output_svg_path = "modality_contribution_accuracy_chart.svg"
plt.savefig(output_svg_path, format="svg")
plt.close()
print("SVG chart saved successfully.")
