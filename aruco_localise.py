# Minimal ArUco pose estimation on Raspberry Pi using Picamera2 + OpenCV
# - Shows live stream with detected marker outlines and axes
# - Prints pose (rvec, tvec) to console for each detected marker
# Requirements: opencv-contrib-python, picamera2

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

# ---- ArUco setup ----
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # change if your dictionary differs
detector_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detector_params)

print("Press 'q' in the window to quit.")
last_ids_printed = set()

try:
    while True:
        frame = picam2.capture_array()  # RGB
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, _rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Draw marker borders and IDs
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for each detected marker
            # (returns rvecs and tvecs with shape (N, 1, 3))
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec = rvecs[i][0], tvecs[i][0]

                # Draw axis (length = half the marker side, for visibility)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS * 0.5)

                # Put a small text label near the marker
                c = corners[i][0].astype(int)
                corner_pt = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                cv2.putText(frame, f"ID {int(marker_id)}", corner_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Print pose to console (in meters; rvec is Rodrigues rotation vector in radians)
                # Only print when new detection or ID changed frame-to-frame to reduce spam
                # (comment out the if-block to print every frame)
                key_for_id = (int(marker_id), round(float(tvec[0]), 3), round(float(tvec[1]), 3), round(float(tvec[2]), 3))
                if key_for_id not in last_ids_printed:
                    print(f"Marker ID {int(marker_id)}:")
                    print(f"  tvec (m): [{tvec[0]: .4f}, {tvec[1]: .4f}, {tvec[2]: .4f}]")
                    print(f"  rvec (rad): [{rvec[0]: .4f}, {rvec[1]: .4f}, {rvec[2]: .4f}]")
                    # Optionally print yaw/pitch/roll (XYZ intrinsic) derived from R
                    R, _ = cv2.Rodrigues(rvec)
                    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                    singular = sy < 1e-6
                    if not singular:
                        x = np.arctan2(R[2, 1], R[2, 2])
                        y = np.arctan2(-R[2, 0], sy)
                        z = np.arctan2(R[1, 0], R[0, 0])
                    else:
                        x = np.arctan2(-R[1, 2], R[1, 1])
                        y = np.arctan2(-R[2, 0], sy)
                        z = 0.0
                    print(f"  Euler XYZ (rad): roll={x: .4f}, pitch={y: .4f}, yaw={z: .4f}")
                    last_ids_printed.add(key_for_id)
        else:
            last_ids_printed.clear()

        # Show stream
        cv2.imshow("ArUco Pose (press 'q' to quit)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
