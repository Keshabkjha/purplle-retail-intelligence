from app.main import ingest_events

__all__ = ["ingest_events"]

if __name__ == "__main__":
    import requests
    print("🌱 Seeding POS transactions from: Brigade_Bangalore_10_April_26 (1)bc6219c.csv...")
    try:
        r = requests.post("http://localhost:8000/api/load-pos")
        data = r.json()
        if r.status_code == 200:
            print(f"✅ Successfully seeded {data.get('loaded', 0)} POS transaction records into tables.")
        else:
            print(f"❌ Failed: {data.get('detail', r.text)}")
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
