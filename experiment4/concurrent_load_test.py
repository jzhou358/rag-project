import argparse
import time
import csv
from datetime import datetime
import statistics
import concurrent.futures
import requests


def do_request(session, url: str, timeout: float = 15.0):
    """Send one HTTP GET request and measure latency."""
    start = time.perf_counter()
    try:
        resp = session.get(url, timeout=timeout)
        latency = time.perf_counter() - start
        return {
            "ok": True,
            "status": resp.status_code,
            "latency": latency,
        }
    except Exception as e:
        latency = time.perf_counter() - start
        return {
            "ok": False,
            "status": None,
            "latency": latency,
            "error": str(e),
        }


def run_load_test(url: str, total_requests: int, concurrency: int, csv_path: str | None):
    print(f"Target URL: {url}")
    print(f"Total requests: {total_requests}")
    print(f"Concurrency (threads): {concurrency}")
    print("-" * 60)

    results = []

    def worker(_):
        with requests.Session() as session:
            return do_request(session, url)

    start_all = time.perf_counter()

    # Use a thread pool to send requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(total_requests)]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if res["ok"]:
                print(f"[{idx}/{total_requests}] OK  status={res['status']}  latency={res['latency']:.3f}s")
            else:
                print(f"[{idx}/{total_requests}] FAIL error={res.get('error', 'unknown')}")

    total_time = time.perf_counter() - start_all
    print("-" * 60)
    print(f"Finished {total_requests} requests in {total_time:.2f} seconds")

    # ----- Aggregate statistics -----
    latencies_ok = [r["latency"] for r in results if r["ok"]]
    successes = sum(1 for r in results if r["ok"])
    failures = total_requests - successes

    if latencies_ok:
        avg_latency = statistics.mean(latencies_ok)
        p95_latency = statistics.quantiles(latencies_ok, n=20)[18]  # approx 95th percentile
    else:
        avg_latency = None
        p95_latency = None

    print(f"Success: {successes}, Fail: {failures}")
    if avg_latency is not None:
        print(f"Average latency: {avg_latency:.3f} s")
        print(f"Approx. 95th percentile latency: {p95_latency:.3f} s")

    # ----- Save to CSV (optional) -----
    if csv_path:
        timestamp = datetime.utcnow().isoformat()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "url",
                "total_requests",
                "concurrency",
                "successes",
                "failures",
                "avg_latency_sec",
                "p95_latency_sec",
            ])
            writer.writerow([
                timestamp,
                url,
                total_requests,
                concurrency,
                successes,
                failures,
                f"{avg_latency:.6f}" if avg_latency is not None else "",
                f"{p95_latency:.6f}" if p95_latency is not None else "",
            ])
        print(f"Saved summary to {csv_path}")

    return {
        "total_time": total_time,
        "successes": successes,
        "failures": failures,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple concurrent load tester for your RAG web app.")
    parser.add_argument(
        "--url",
        required=True,
        help="Target URL (e.g. http://your-alb-dns-name/ )",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests to send (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent worker threads (default: 10)",
    )
    parser.add_argument(
        "--csv",
        default="load_test_summary.csv",
        help="Output CSV file for summary statistics (default: load_test_summary.csv, empty string to disable)",
    )

    args = parser.parse_args()

    csv_path = args.csv if args.csv.strip() else None

    run_load_test(
        url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()