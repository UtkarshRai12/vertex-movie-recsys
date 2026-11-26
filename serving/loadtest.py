# Time /recommend for users 1..200 after a short warmup.
# Run: python serving/loadtest.py <URL>
import subprocess
import sys
import time

import requests

url = sys.argv[1].rstrip("/")
token = subprocess.check_output(
    ["gcloud", "auth", "print-identity-token"], text=True
).strip()
headers = {"Authorization": f"Bearer {token}"}


def one(uid: int):
    started = time.perf_counter()
    response = requests.get(f"{url}/recommend/{uid}", headers=headers, timeout=60)
    elapsed = (time.perf_counter() - started) * 1000
    return response.status_code, elapsed


# The first Cloud Run call includes container startup. Do not count it.
for uid in (1, 2, 3):
    code, elapsed = one(uid)
    print(f"warmup {uid} {code} {elapsed:.0f} ms", flush=True)

latencies = []
skipped = []
for uid in range(1, 201):
    code, elapsed = one(uid)
    if code == 404:
        skipped.append(uid)
        continue
    if code != 200:
        raise SystemExit(f"user {uid} returned {code}")
    latencies.append(elapsed)

latencies.sort()
count = len(latencies)
p50 = latencies[int(0.50 * (count - 1))]
p95 = latencies[int(0.95 * (count - 1))]
print(f"n = {count} skipped = {skipped}")
print(f"p50 = {p50:.0f} ms  p95 = {p95:.0f} ms")
