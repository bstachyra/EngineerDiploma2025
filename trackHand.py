# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv # Added for saving data
import datetime # Added for timestamping filename

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# `max_num_hands`: Maximum number of hands to detect.
# `min_detection_confidence`: Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
# `min_tracking_confidence`: Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be considered tracked successfully.
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
# Use 0 for the default webcam. Change if you have multiple cameras.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Store the path points. Using deque for potentially limiting path length later if needed.
# We'll track the index fingertip (landmark 8).
path_points = deque(maxlen=1024) # Store up to 1024 points

# Create a blank canvas to draw the persistent path
path_canvas = None

# --- New variables ---
is_tracking = True # Flag to control tracking state
tracking_status_text = "Tracking ON"
tracking_status_color = (0, 255, 0) # Green for ON

print("Starting Hand Tracking...")
print("Press 's' to start/stop tracking.")
print("Press 'c' to clear the path.")
print("Press 'w' to write/save the current path to a CSV file.")
print("Press 'q' to quit.")

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Get frame dimensions
    h, w, _ = frame.shape

    # Initialize the path canvas on the first frame or after clearing
    if path_canvas is None:
        path_canvas = np.zeros_like(frame)

    # Flip the frame horizontally for a later selfie-view display
    # Also convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # Process hand detection only if tracking is enabled
    center_x, center_y = None, None # Initialize coordinates for this frame
    if is_tracking:
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Draw the hand annotations on the image if hands are detected
        if results.multi_hand_landmarks:
            # Process only the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks and connections on the RGB frame before converting back
            mp_drawing.draw_landmarks(
                frame_rgb, # Draw on the RGB frame
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index fingertip (landmark 8)
            index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized coordinates (0.0-1.0) to pixel coordinates
            center_x = int(index_fingertip.x * w)
            center_y = int(index_fingertip.y * h)

            # Append the center point to the path list only if tracking
            path_points.appendleft((center_x, center_y))
    else:
        # If not tracking, ensure we don't add points from previous detections
        pass # No processing needed if tracking is off

    # Convert the potentially annotated RGB frame back to BGR for OpenCV rendering
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Draw the path on the frame and the canvas regardless of tracking status,
    # but points are only added when tracking is ON.
    # Iterate through the points stored in the deque
    if len(path_points) > 1: # Need at least two points to draw a line
        for i in range(1, len(path_points)):
            # If either of the points is None (shouldn't happen with current logic, but good practice)
            if path_points[i - 1] is None or path_points[i] is None:
                continue
            # Draw a line between consecutive points on the frame
            cv2.line(frame, path_points[i - 1], path_points[i], (0, 255, 0), 2) # Green line on live frame
            # Draw the same line on the persistent canvas
            cv2.line(path_canvas, path_points[i - 1], path_points[i], (255, 50, 50), 2) # Blue line on canvas

    # Combine the frame with the path canvas
    # Use addWeighted to blend the path canvas onto the current frame
    frame = cv2.addWeighted(frame, 0.8, path_canvas, 0.2, 0)

    # --- Display Tracking Status ---
    cv2.putText(frame, tracking_status_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_status_color, 2)

    # Display the resulting frame
    cv2.imshow('Hand Tracking Path', frame)

    # Handle key presses
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        # Quit the loop if 'q' is pressed
        break
    elif key == ord('c'):
        # Clear the path if 'c' is pressed
        path_points.clear()
        path_canvas = np.zeros_like(frame) # Reset the canvas
        print("Path cleared.")
    elif key == ord('s'):
        # Toggle tracking state if 's' is pressed
        is_tracking = not is_tracking
        if is_tracking:
            tracking_status_text = "Tracking ON"
            tracking_status_color = (0, 255, 0) # Green
            print("Tracking resumed.")
        else:
            tracking_status_text = "Tracking OFF"
            tracking_status_color = (0, 0, 255) # Red
            print("Tracking paused.")
    elif key == ord('w'):
        # Save path data if 'w' is pressed
        if not path_points:
            print("No path data to save.")
        else:
            # Generate a filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hand_path_{timestamp}.csv"
            try:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['x', 'y']) # Write header
                    # Iterate from oldest to newest point (deque stores newest at index 0)
                    for point in reversed(path_points):
                        if point: # Ensure point is not None
                           writer.writerow([point[0], point[1]])
                print(f"Path saved successfully to {filename}")
            except Exception as e:
                print(f"Error saving path: {e}")


# Release the webcam and destroy all OpenCV windows
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Application finished.")
    