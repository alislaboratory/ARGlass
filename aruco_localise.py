#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import sys
from math import atan2, asin, degrees

def load_calibration_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Could not open calibration file: {path}")
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("distortion_coefficients").mat()
    fs.release()
    if K is None or D is None:
        raise ValueError("camera_matrix or distortion_coefficients not found in YAML")
    D = D.reshape(1, -1)  # ensure shape (1,N)
    return K, D

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def invert_T(T):
    Ri = T[:3,:3].T
    ti = -Ri @ T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = Ri
    Ti[:3,3] = ti
    return Ti

def R_to_ypr_zyx(R):
    # ZYX convention (yaw around Z, pitch around Y, roll around X)
    yaw = degrees(atan2(R[1,0], R[0,0]))
    pitch = degrees(asin(-R[2,0]))
    roll = degrees(atan2(R[2,1], R[2,2]))
    return yaw, pitch, roll

# Map friendly names to OpenCV constants
DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_50": cv2.aruco.DICT_7X7_50,
    "7X7_100": cv2.aruco.DICT_7X7_100,
    "7X7_250": cv2.aruco.DICT_7X7_250,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
    "APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

def get_aruco_dict(dict_name):
    # Works across older/newer OpenCV builds
    const = DICT_MAP[dict_name]
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(const)
    else:
        return cv2.aruco.Dictionary_get(const)

def main():
    ap = argparse.ArgumentParser(description="Print camera pose relative to an ArUco tag (legacy OpenCV API).")
    ap.add_argument("--calib", default="camera_calibration.yaml",
                    help="Path to OpenCV YAML with camera_matrix and distortion_coefficients.")
    ap.add_argument("--marker-length", type=float, required=True,
                    help="Marker edge length in meters (e.g., 0.04 for 4 cm).")
    ap.add_argument("--dict", default="4X4_50", choices=list(DICT_MAP.keys()),
                    help="ArUco dictionary for your printed tag.")
    ap.add_argument("--id", type=int, default=None,
                    help="If set, only report this marker ID (ignore others).")
    ap.add_argument("--camera-index", type=int, default=0,
                    help="cv2.VideoCapture index (default 0).")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    # Load calibration
    try:
        K, D = load_calibration_yaml(args.calib)
    except Exception as e:
        print(f"[ERROR] Loading calibration: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare detector (legacy API)
    try:
        aruco_dict = get_aruco_dict(args.dict)
        params = (cv2.aruco.DetectorParameters_create()
                  if hasattr(cv2.aruco, "DetectorParameters_create")
                  else cv2.aruco.DetectorParameters())  # very old builds may expose class ctor
    except Exception as e:
        print(f"[ERROR] ArUco setup failed: {e}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.", file=sys.stderr)
        sys.exit(1)

    print("# Running. Ctrl+C to quit.", flush=True)
    print("# Output: id, x y z (m), yaw pitch roll (deg)  [camera in tag frame]", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame.")
                continue

            # Detect markers (legacy API)
            corners, ids, _rej = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

            if ids is None or len(ids) == 0:
                continue

            # Estimate pose
            rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                corners, args.marker_length, K, D
            )

            for i, id_ in enumerate(ids.flatten()):
                if args.id is not None and id_ != args.id:
                    continue

                rvec = rvecs[i]
                tvec = tvecs[i]

                # tag -> camera
                T_tag_in_cam = rvec_tvec_to_T(rvec, tvec)
                # camera -> tag (what we want)
                T_cam_in_tag = invert_T(T_tag_in_cam)

                t = T_cam_in_tag[:3, 3]
                R = T_cam_in_tag[:3, :3]
                yaw, pitch, roll = R_to_ypr_zyx(R)

                print(f"id {id_}: "
                      f"x={t[0]:+.3f}  y={t[1]:+.3f}  z={t[2]:+.3f}  "
                      f"yaw={yaw:+.1f}  pitch={pitch:+.1f}  roll={roll:+.1f}",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

if __name__ == "__main__":
    main()
