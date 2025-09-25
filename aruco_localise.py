# ArUco pose estimation (OpenCV 4.6.0) using Picamera2 + OpenCV
# - Shows live stream with detected marker outlines and axes
# - Prints pose (rvec, tvec) to console for each detected marker
# Fixes: drawDetectedMarkers assertion by always drawing on a 3-channel BGR image.
# Requirements: opencv-contrib-python==4.6.0.*, picamera2

import cv2
import numpy as np
from picamera2 import Picamera2

MARKER_LENGTH_METERS = 0.094  # <- set to your tag's side length (m)

# --- Load calibration (OpenCV YAML) ---
calib_path = "calibration/camera_calibration.yaml"
fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise FileNotFoundError(f"Could not open calibration file: {calib_path}")
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("distortion_coefficients").mat()
fs.release()
if camera_matrix is None or dist_coeffs is None:
    raise ValueError("Failed to load camera_matrix or distortion_coefficients from YAML.")

# --- Picamera2 setup ---
picam2 = Picamera2()
# picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)}))
picam2.start()

# --- ArUco (OpenCV 4.6.0 API) ---
aruco = cv2.aruco
dictionary = aruco.Dictionary_get(aruco.DICT_6X6_50)  # change if your tags use another dict
parameters = aruco.DetectorParameters_create()

print("Press 'q' to quit.")

try:
    while True:
        frame = picam2.capture_array()  # RGB (uint8)

        # Defensive: ensure we have a valid frame
        if frame is None or frame.size == 0:
            continue

        # Create a guaranteed 3-channel BGR image for drawing (avoids assertion error)
        if frame.ndim == 3 and frame.shape[2] == 3:
            draw_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            draw_img = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            # If somehow single-channel, convert to BGR for drawing
            draw_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Use grayscale for detection
        gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)

        corners, ids, _rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Draw marker borders/IDs on the 3-channel image
            aruco.drawDetectedMarkers(draw_img, corners, ids)

            # Pose estimation
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec = rvecs[i][0], tvecs[i][0]

                # Draw axis
                cv2.drawFrameAxes(draw_img, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS * 0.5)

                # Label near the marker center
                c = corners[i][0].astype(int)
                center_pt = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                cv2.putText(draw_img, f"ID {int(marker_id)}", center_pt,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Print pose (meters) to console
                print(f"ID {int(marker_id)}  tvec(m)=[{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}]  "
                      f"rvec(rad)=[{rvec[0]:.4f}, {rvec[1]:.4f}, {rvec[2]:.4f}]")

        cv2.imshow("ArUco Pose (OpenCV 4.6.0) - press 'q' to quit", draw_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
