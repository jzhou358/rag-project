# plot_latency_comparison.py

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load the CSV
    df = pd.read_csv("latency_results.csv")

    # Sanity check column names (adapt if yours are slightly different)
    # Expected columns: "question", "mode", "runs", "avg_latency_sec"
    expected_cols = {"question", "mode", "runs", "avg_latency_sec"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 2. Aggregate by mode
    #   - mean latency per mode
    #   - std deviation (for error bars)
    summary = (
        df.groupby("mode")["avg_latency_sec"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )

    print("Aggregated latency stats (seconds):")
    print(summary)

    # 3. Plot bar chart: GPT-only vs RAG
    modes = summary["mode"]
    means = summary["mean"]
    stds = summary["std"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(modes, means, yerr=stds, capsize=5)
    ax.set_ylabel("Average Latency (seconds)")
    ax.set_xlabel("Mode")
    ax.set_title("End-to-End Latency Comparison: GPT-Only vs RAG")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Optional: show exact values on top of bars
    for x, y in zip(modes, means):
        ax.text(x, y + 0.05, f"{y:.2f}s", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("latency_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
