import argparse
import os

import cv2

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")
        # Draw a small circle where clicked
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', img)

def main(image_or_video_path):
    global img
    
    if not os.path.exists(image_or_video_path):
        print(f"File not found: {image_or_video_path}")
        return

    # Check if it's a video
    if image_or_video_path.endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(image_or_video_path)
        ret, img = cap.read()
        cap.release()
        if not ret:
            print("Failed to read video frame.")
            return
    else:
        # It's an image
        img = cv2.imread(image_or_video_path)

    # Resize if too large to fit on screen
    h, w = img.shape[:2]
    max_height = 800
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), max_height))
        print(f"Image resized to fit screen. Scale factor: {scale}")
        print("Note: Clicked coordinates will be based on the scaled image. For true calibration, you might need to multiply the points by the inverse scale.")

    cv2.imshow('image', img)
    print("--------------------------------------------------")
    print(f"Loaded: {image_or_video_path}")
    print("Click exactly 4 points on the floor to form a rectangle/polygon.")
    print("Press 'q' when done, or 'c' to clear points.")
    print("--------------------------------------------------")

    cv2.setMouseCallback('image', click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            points.clear()
            # Reload image to clear dots
            if image_or_video_path.endswith('.mp4'):
                cap = cv2.VideoCapture(image_or_video_path)
                _, img = cap.read()
                cap.release()
            else:
                img = cv2.imread(image_or_video_path)
            if h > max_height:
                img = cv2.resize(img, (int(w * scale), max_height))
            cv2.imshow('image', img)
            print("Points cleared.")

    cv2.destroyAllWindows()
    print("Final points selected:", points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera points")
    parser.add_argument("path", help="Path to video or image file")
    args = parser.parse_args()
    main(args.path)
