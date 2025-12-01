import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "codebuild_build_time.csv"
OUTPUT_FIG = "codepipeline_build_time.png"


def main():
    # 1. Load data
    df = pd.read_csv(CSV_PATH)

    # Ensure it is sorted by time
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add a simple run index (Run 1, Run 2, ...)
    df["run"] = range(1, len(df) + 1)

    # Optional: convert seconds to minutes to make the numbers nicer
    df["duration_min"] = df["duration_sec"] / 60.0

    # 2. Plot
    plt.figure(figsize=(10, 5))

    plt.bar(df["run"], df["duration_min"], width=0.6)

    # Annotate each bar with the exact time
    for x, y in zip(df["run"], df["duration_min"]):
        plt.text(
            x,
            y + 0.05,
            f"{y:.1f} min",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Build Run")
    plt.ylabel("Build Duration (minutes)")
    plt.title("CI/CD Build Duration per CodePipeline/CodeBuild Run")
    plt.xticks(df["run"], [f"Run {r}" for r in df["run"]])

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300)
    plt.show()

    print(f"Saved figure to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
