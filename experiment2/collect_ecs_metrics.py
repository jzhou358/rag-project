import boto3
import datetime
import csv

# --- CONFIG ---
REGION = "us-east-1"             # your region
CLUSTER_NAME = "rag-project-cluster"
SERVICE_NAME = "rag-service"
PERIOD = 60                      # 60 seconds per data point
OUTPUT_CSV = "ecs_cpu_memory_usage.csv"


END_TIME = datetime.datetime.utcnow()
START_TIME = END_TIME - datetime.timedelta(minutes=30)  # last 30 minutes

cloudwatch = boto3.client("cloudwatch", region_name=REGION)

def get_metric(metric_name, stat="Average"):
    resp = cloudwatch.get_metric_statistics(
        Namespace="AWS/ECS",
        MetricName=metric_name,
        Dimensions=[
            {"Name": "ClusterName", "Value": CLUSTER_NAME},
            {"Name": "ServiceName", "Value": SERVICE_NAME},
        ],
        StartTime=START_TIME,
        EndTime=END_TIME,
        Period=PERIOD,
        Statistics=[stat],
    )
    
    datapoints = sorted(resp["Datapoints"], key=lambda x: x["Timestamp"])
    return datapoints

def main():
    cpu_data = get_metric("CPUUtilization")
    mem_data = get_metric("MemoryUtilization")

    
    rows = []
    for c, m in zip(cpu_data, mem_data):
        timestamp = c["Timestamp"].isoformat()
        cpu = c["Average"]
        mem = m["Average"]
        rows.append((timestamp, cpu, mem))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cpu_utilization", "memory_utilization"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
