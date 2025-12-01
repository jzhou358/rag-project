import pandas as pd
import matplotlib.pyplot as plt

# Load your collected metrics
df = pd.read_csv("ecs_cpu_memory_usage.csv")

# Create a simple index for x-axis labels
df['index'] = range(1, len(df) + 1)

plt.figure(figsize=(16, 6))

plt.plot(df['index'], df['cpu_utilization'], label="CPU Utilization (%)", marker='o')
plt.plot(df['index'], df['memory_utilization'], label="Memory Utilization (%)", marker='o')

plt.xlabel("Time Interval (T1, T2, T3, ...)")
plt.ylabel("Usage (%)")
plt.title("ECS CPU and Memory Utilization Over Time")

# Set the ticks to show T1, T2, T3...
plt.xticks(df['index'], [f"T{i}" for i in df['index']], rotation=0)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("ecs_metrics_clean.png", dpi=300)
plt.show()
