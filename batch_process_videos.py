import os
import subprocess

search_dirs = ["Store 1", "Store 2"]

for d in search_dirs:
    for filename in os.listdir(d):
        if filename.endswith(".mp4"):
            filepath = os.path.join(d, filename)
            annotated_path = f"annotated_{filename}"
            if not os.path.exists(annotated_path):
                print(f"Processing {filepath}...")
                env = os.environ.copy()
                env["PYTHONPATH"] = os.getcwd()
                subprocess.run(["python3", "pipeline/detect.py", filepath], env=env)
            else:
                print(f"Already processed: {annotated_path}")

print("All videos processed.")
