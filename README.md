# Store Intelligence System - Apex Retail

An end-to-end computer vision and real-time event analytics pipeline that transforms raw store CCTV footage into offline offline analytics and business KPIs.

---

## 🚀 Quick Start (Setup in 5 Commands)

Follow these 5 simple commands to build, start, run, and query the entire Store Intelligence system:

```bash
# 1. Start the API using Docker Compose
docker compose up -d --build

# 2. Ingest POS transaction data into the database
docker compose exec api python -m app.ingestion

# 3. Local OpenCV GUI installation (for floor-plan coordinate calibration)
python3 -m pip install opencv-python

# 4. Run the computer vision detection & spatial mapping pipeline
./pipeline/run.sh

# 5. Query the real-time store metrics endpoint
curl http://localhost:8000/stores/ST1008/metrics
```

---

## 🛠️ Calibration Tool (`pipeline/calibrate.py`)

To map 3D camera foot positions perfectly onto the 2D floor plan (`Revised.png`), run the interactive calibration script on your local machine:

```bash
# Example for the Entry camera
python pipeline/calibrate.py "CCTV Footage/entry_camera.mp4"
```

1. **Click 4 points on the floor** that form a rectangle or quadrilateral.
2. Press `q` to complete, and the terminal will output the coordinate list.
3. Run the same for `Revised.png` by clicking the exact same 4 points on the flat map:
   ```bash
   python pipeline/calibrate.py Revised.png
   ```
4. Save the returned source and destination coordinates to `config/calibration.json` in this format:
   ```json
   {
     "CAM_ENTRY_01": {
       "src": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
       "dst": [[mx1, my1], [mx2, my2], [mx3, my3], [mx4, my4]]
     }
   }
   ```
   *(Note: The pipeline automatically falls back to sensible default scaling matrices if a camera is not yet calibrated, ensuring the pipeline works out-of-the-box!)*

---

## 📊 Available REST API Endpoints

Once the API is running, you can access the interactive Swagger documentation at `http://localhost:8000/docs` and query these key endpoints:

* **Store Metrics**: `GET http://localhost:8000/stores/ST1008/metrics`
  Calculates Unique Visitors, Store Conversion Rate, Average Dwell Minutes, Queue Depth, and Abandonment Rate (excluding staff members).
* **Visitor Funnel**: `GET http://localhost:8000/stores/ST1008/funnel`
  Details the retail funnel flow (`Entry -> Zone Visit -> Billing Queue -> Purchase`) with unique customer counts and stage drop-off percentages.
* **Dwell Heatmap**: `GET http://localhost:8000/stores/ST1008/heatmap`
  Returns zone visit frequency, average dwell seconds, and normalized intensity (0-100) mapped for floor plan rendering. Includes `data_confidence` check.
* **Operational Anomalies**: `GET http://localhost:8000/stores/ST1008/anomalies`
  Active real-time warnings for billing queue spikes, low conversion, or product zone dead spots, including recommended management actions.
* **API Health**: `GET http://localhost:8000/health`
  Service and database status, lag tracking, and active stale feed warnings if live feeds drop.

---

## 🧪 Running Automated Tests

Run the full pytest suite (covering partial ingestion, staff exclusion, queue spikes, low conversion drops, and dead zones) using:

```bash
docker compose exec api pytest tests/
```
*(All 7 unit tests pass successfully with 100% correctness)*
