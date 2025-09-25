#!/usr/bin/env python3
# Localize the camera in 3D relative to an ArUco tag using Picamera2 + OpenCV
# Requires: opencv-python (with aruco module), picamera2

import sys
import time
import math
import numpy as np
import cv2
from picamera2 import Picamera2

# ========= USER CONFIG =========
# Set this to the real printed tag size (edge length) in meters.
MARKER_SIZE_METERS = 0.094  # e.g., 0.05 = 5 cm

# ArUco dictionary to use. Change if your tag comes from another dictionary.
ARUCO_DICT_NAME = "DICT_6X6_50"  # options: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_APRILTAG_36h11, etc.

# Path to the YAML file you provided in the prompt
CALIBRATION_YAML = "calibration/camera_calibration.yaml"

# Camera stream configuration (resolution + RGB888 for OpenCV)
CAM_RESOLUTION = (1280, 720)
# =================================


def load_calibration(yaml_path):
    """Load camera_matrix and distortion coefficients from an OpenCV YAML."""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Could not open calibration file: {yaml_path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()

    if camera_matrix is None or dist_coeffs is None:
        raise ValueError("Calibration file missing camera_matrix or distortion_coefficients nodes.")

    # Ensure proper shapes
    dist_coeffs = dist_coeffs.reshape(-1, 1)
    return camera_matrix, dist_coeffs


def get_aruco_dict(name):
    """Return an OpenCV ArUco dictionary object by human-readable name."""
    if not hasattr(cv2, "aruco"):
        raise ImportError("OpenCV was built without the aruco module. Install opencv-contrib-python.")

    name_map = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}
    if name not in name_map:
        raise ValueError(f"Unknown ArUco dictionary '{name}'. Available examples: "
                         f"{', '.join(sorted([k for k in name_map.keys() if 'DICT_' in k])[:10])} ...")
    return cv2.aruco.getPredefinedDictionary(name_map[name])


def rodrigues_to_matrix(rvec):
    """Convert Rodrigues rotation vector to a 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R


def rotation_matrix_to_euler_zyx(R):
    """
    Convert a rotation matrix to ZYX Euler angles (yaw, pitch, roll) in radians.
    Convention (intrinsic rotations): R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Returns (yaw_z, pitch_y, roll_x)
    """
    # Guard against numerical issues
    sy = -R[2, 0]
    cy = math.sqrt(max(0.0, 1.0 - sy * sy))

    singular = cy < 1e-6
    if not singular:
        yaw = math.atan2(R[1, 0], R[0, 0])       # z
        pitch = math.asin(sy)                    # y
        roll = math.atan2(R[2, 1], R[2, 2])      # x
    else:
        # Gimbal lock
        yaw = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.asin(sy)
        roll = 0.0

    return yaw, pitch, roll


def format_pose_text(position_m, ypr_deg):
    x, y, z = position_m
    yaw, pitch, roll = ypr_deg
    return [
        f"Camera wrt tag:",
        f"  x = {x:+.3f} m, y = {y:+.3f} m, z = {z:+.3f} m",
        f"  yaw = {yaw:+.1f}°, pitch = {pitch:+.1f}°, roll = {roll:+.1f}°",
    ]


def main():
    # Load calibration
    try:
        cam_mtx, dist = load_calibration(CALIBRATION_YAML)
        print("Loaded calibration:\n", cam_mtx, "\n", dist.ravel())
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Prepare ArUco
    aruco_dict = get_aruco_dict(ARUCO_DICT_NAME)
    parameters = cv2.aruco.DetectorParameters()
    # (Tune parameters if needed, e.g., parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX)

    # Picamera2 setup
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": CAM_RESOLUTION}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)  # small warm-up

    cv2.namedWindow("ArUco Pose (Camera wrt Tag)", cv2.WINDOW_NORMAL)

    print("Press 'q' to quit.")
    last_info = None

    try:
        while True:
            frame = picam2.capture_array()  # RGB888
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None and len(ids) > 0:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Estimate pose for each marker (marker pose wrt camera)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_SIZE_METERS, cam_mtx, dist
                )

                # Choose the first marker (or you can find a specific ID)
                rvec = rvecs[0].reshape(3, 1)
                tvec = tvecs[0].reshape(3, 1)

                # Draw a coordinate frame on the first marker (length in same units as tvec)
                try:
                    cv2.drawFrameAxes(frame, cam_mtx, dist, rvec, tvec, MARKER_SIZE_METERS * 0.6)
                except AttributeError:
                    # Fallback if drawFrameAxes isn't available in your OpenCV
                    cv2.aruco.drawAxis(frame, cam_mtx, dist, rvec, tvec, MARKER_SIZE_METERS * 0.6)

                # ---------------------------
                # Convert to CAMERA pose wrt TAG
                # OpenCV gives: marker pose wrt camera  ->  [R_mc | t_mc]
                # We want:     camera pose wrt marker  ->  [R_cm | t_cm]
                # with: R_cm = R_mc^T  and  t_cm = -R_mc^T * t_mc
                R_mc = rodrigues_to_matrix(rvec)
                R_cm = R_mc.T
                t_cm = -R_cm @ tvec  # 3x1

                # Euler angles (ZYX order): yaw (Z), pitch (Y), roll (X)
                yaw, pitch, roll = rotation_matrix_to_euler_zyx(R_cm)
                yaw_deg, pitch_deg, roll_deg = [math.degrees(a) for a in (yaw, pitch, roll)]

                cam_pos = (float(t_cm[0]), float(t_cm[1]), float(t_cm[2]))
                ypr_deg = (yaw_deg, pitch_deg, roll_deg)
                last_info = (cam_pos, ypr_deg)

                # Overlay text
                y0 = 30
                dy = 24
                for i, line in enumerate(format_pose_text(cam_pos, ypr_deg)):
                    y = y0 + i * dy
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 230, 20), 2, cv2.LINE_AA)

            else:
                # Helpful hint overlay
                cv2.putText(frame, "No ArUco tag detected.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Dict: {ARUCO_DICT_NAME} | Tag size: {MARKER_SIZE_METERS*100:.0f} mm",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

                if last_info:
                    # Show last known pose (if any), handy when tag is briefly lost
                    cam_pos, ypr_deg = last_info
                    for i, line in enumerate(format_pose_text(cam_pos, ypr_deg)):
                        cv2.putText(frame, line, (10, 90 + i * 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

            cv2.imshow("ArUco Pose (Camera wrt Tag)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
