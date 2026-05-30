#!/bin/bash
# One command to run the computer vision detection pipeline over all CCTV footage clips.

echo "========================================================="
echo "🚀 Starting Store Intelligence CCTV Processing Pipeline"
echo "========================================================="

VIDEOS_DIR="CCTV Footage"

# Run detection on each video file in sequence
videos=(
    "$VIDEOS_DIR/entry_camera.mp4"
    "$VIDEOS_DIR/main_floor_1.mp4"
    "$VIDEOS_DIR/main_floor_2.mp4"
    "$VIDEOS_DIR/main_floor_3.mp4"
    "$VIDEOS_DIR/billing_camera.mp4"
)

for video in "${videos[@]}"; do
    if [ -f "$video" ]; then
        echo "🎥 Processing: $video..."
        python3 pipeline/detect.py "$video"
        echo "---------------------------------------------------------"
    else
        echo "⚠️ Warning: Video file not found: $video"
    fi
done

echo "========================================================="
echo "✅ Finished processing all available CCTV video clips."
echo "========================================================="
