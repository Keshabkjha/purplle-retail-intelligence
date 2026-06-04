import requests
import json

events = []
with open('sample_eventsbe42122.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            events.append(json.loads(line))

print(f"Loaded {len(events)} events, sending to /events/ingest in batches of 100...")

for i in range(0, len(events), 100):
    batch = events[i:i+100]
    r = requests.post("http://localhost:8000/events/ingest", json=batch)
    print(f"Batch {i}: {r.status_code}")

print("Uploading POS data...")
with open("POS - sample transactionsb1e826f.csv", "rb") as f:
    r = requests.post("http://localhost:8000/api/load-pos", files={"file": f})
    print("POS Upload:", r.status_code, r.json())
