import time
import random
import concurrent.futures
import requests

ALB_URL = "http://rag-project-alb-977672821.us-east-1.elb.amazonaws.com/"  
DURATION_SEC = 300   
CONCURRENCY = 10     

def hit_once():
    """Send one simple GET request to the app."""
    try:
        # For Streamlit, just hitting "/" is enough to trigger the app.
        resp = requests.get(ALB_URL, timeout=10)
        return resp.status_code
    except Exception as e:
        print("Request error:", e)
        return None

def worker(stop_time):
    """Loop until stop_time, sending requests."""
    while time.time() < stop_time:
        status = hit_once()
        if status is not None:
            print("Status:", status)
        # Optional: small random sleep to simulate more realistic traffic
        time.sleep(random.uniform(0.1, 0.5))

def main():
    stop_time = time.time() + DURATION_SEC
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(worker, stop_time) for _ in range(CONCURRENCY)]
        for f in concurrent.futures.as_completed(futures):
            _ = f.result()
    print("Load test finished.")

if __name__ == "__main__":
    main()
