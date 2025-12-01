import csv
import datetime as dt

import boto3

# ---- CONFIG ----
REGION = "us-east-1"                 # change if your CodeBuild is in another region
PROJECT_NAME = "rag-project-build"   # your CodeBuild project name
OUTPUT_CSV = "codebuild_build_time.csv"
HOURS_BACK = 24                      # how far back to look for builds
# -----------------


def main():
    cloudwatch = boto3.client("cloudwatch", region_name=REGION)

    end_time = dt.datetime.utcnow()
    start_time = end_time - dt.timedelta(hours=HOURS_BACK)

    # Duration metric is in seconds
    resp = cloudwatch.get_metric_statistics(
        Namespace="AWS/CodeBuild",
        MetricName="Duration",
        Dimensions=[
            {"Name": "ProjectName", "Value": PROJECT_NAME},
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=60,              # 1-minute granularity is fine
        Statistics=["Average"],
    )

    datapoints = resp["Datapoints"]
    if not datapoints:
        print("No data points found – did you trigger any builds in this time window?")
        return

    # Sort by timestamp so runs are in time order
    datapoints.sort(key=lambda x: x["Timestamp"])

    print(f"Writing {len(datapoints)} rows to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "duration_sec"])
        for dp in datapoints:
            ts = dp["Timestamp"].isoformat()
            duration_sec = dp["Average"]
            writer.writerow([ts, duration_sec])

    print("Done.")


if __name__ == "__main__":
    main()
