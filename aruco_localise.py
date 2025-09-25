# Minimal ArUco pose estimation (OpenCV 4.6.0) using Picamera2 + OpenCV
# - Shows live stream with detected marker outlines and axes
# - Prints pose (rvec, tvec) to console for each detected marker
# Requirements: opencv-contrib-python==4.6.0.*, picamera2

import cv2
import numpy as np
from picamera2 import Picamera2

# ---- USER TWEAKABLE: real-world marker side length (meters) ----
MARKER_LENGTH_METERS = 0.05  # change to your marker's size

# ---- Load camera calibration (OpenCV YAML format) ----
calib_path = "calibration/camera_calibration.yaml"
fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise FileNotFoundError(f"Could not open calibration file: {calib_path}")
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("distortion_coefficients").mat()
fs.release()
if camera_matrix is None or dist_coeffs is None:
    raise ValueError("Failed to load camera_matrix or distortion_coefficients from YAML.")

# ---- Set up Picamera2 ----
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)}))
picam2.start()

# ---- ArUco setup for OpenCV 4.6.0 ----
aruco = cv2.aruco
dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)  # change if your dictionary differs
parameters = aruco.DetectorParameters_create()

print("Press 'q' in the window to quit.")

try:
    while True:
        frame = picam2.capture_array()  # RGB
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 4.6.0-style detection
        corners, ids, _rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Draw marker borders and IDs
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for each detected marker
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec = rvecs[i][0], tvecs[i][0]

                # Draw axis (length = half the marker side)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS * 0.5)

                # Label
                c = corners[i][0].astype(int)
                corner_pt = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                cv2.putText(frame, f"ID {int(marker_id)}", corner_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Print pose to console (meters; rvec is Rodrigues rotation vector in radians)
                print(f"Marker ID {int(marker_id)}:")
                print(f"  tvec (m): [{tvec[0]: .4f}, {tvec[1]: .4f}, {tvec[2]: .4f}]")
                print(f"  rvec (rad): [{rvec[0]: .4f}, {rvec[1]: .4f}, {rvec[2]: .4f}]")

        # Show stream
        cv2.imshow("ArUco Pose (press 'q' to quit)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
