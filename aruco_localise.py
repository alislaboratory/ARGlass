#!/usr/bin/env python3
# Robust ArUco camera pose (Picamera2 + OpenCV), avoids drawFrameAxes segfaults
import os, sys, time, math, argparse
import numpy as np
import cv2
from picamera2 import Picamera2

# ===== USER CONFIG =====
MARKER_SIZE_METERS = 0.05       # edge length of your marker in meters
ARUCO_DICT_NAME = "DICT_4X4_50" # change if your tag uses a different dict
CALIBRATION_YAML = "calibration/camera_calibration.yaml"
CAM_RESOLUTION = (1280, 720)
WINDOW_TITLE = "ArUco Pose (Camera wrt Tag)"
# =======================

# ---------- utils ----------
def load_calibration(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Could not open calibration file: {yaml_path}")
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("distortion_coefficients").mat()
    fs.release()
    if K is None or D is None:
        raise ValueError("Calibration file missing camera_matrix or distortion_coefficients.")
    return K, D.reshape(-1, 1)

def get_aruco_dict(name):
    if not hasattr(cv2, "aruco"):
        raise ImportError("This OpenCV build lacks cv2.aruco (install opencv-contrib).")
    constants = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}
    if name not in constants:
        raise ValueError(f"Unknown ArUco dict '{name}'. Examples: {', '.join(sorted(constants)[:8])} ...")
    return cv2.aruco.getPredefinedDictionary(constants[name])

def make_detector(aruco_dict):
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, params), True
    return (aruco_dict, params), False

def detect(detector, use_new_api, gray):
    if use_new_api:
        return detector.detectMarkers(gray)  # corners, ids, rejected
    d, p = detector
    return cv2.aruco.detectMarkers(gray, d, parameters=p)

def to_euler_zyx(R):
    sy = -R[2, 0]
    cy = max(0.0, 1.0 - sy*sy) ** 0.5
    if cy >= 1e-6:
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.asin(sy)
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        yaw = math.atan2(-R[0, 1], R[1, 1]); pitch = math.asin(sy); roll = 0.0
    return yaw, pitch, roll

def draw_axes_manual(img, K, D, rvec, tvec, axis_len):
    # Draw X (red), Y (green), Z (blue) axes from marker origin using projectPoints
    origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
    axes = np.float32([[axis_len,0,0], [0,axis_len,0], [0,0,axis_len]]).reshape(-1, 3)
    pts_o, _ = cv2.projectPoints(origin, rvec, tvec, K, D)
    pts_a, _ = cv2.projectPoints(axes,  rvec, tvec, K, D)
    o = tuple(np.int32(pts_o[0,0]))
    x = tuple(np.int32(pts_a[0,0])); y = tuple(np.int32(pts_a[1,0])); z = tuple(np.int32(pts_a[2,0]))
    # use default colors; if your build crashes with colored lines, remove color tuples
    cv2.line(img, o, x, (0,0,255), 2)   # X
    cv2.line(img, o, y, (0,255,0), 2)   # Y
    cv2.line(img, o, z, (255,0,0), 2)   # Z

def format_pose_text(t_cm, ypr_deg):
    x,y,z = t_cm; yaw,pitch,roll = ypr_deg
    return [
        "Camera wrt tag:",
        f"  x={x:+.3f} m  y={y:+.3f} m  z={z:+.3f} m",
        f"  yaw={yaw:+.1f}°  pitch={pitch:+.1f}°  roll={roll:+.1f}°",
    ]

def safe_imshow(title, img):
    try:
        cv2.imshow(title, img)
        return True
    except Exception:
        return False

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="Print poses; no GUI window.")
    ap.add_argument("--no-overlay", action="store_true", help="Skip drawing markers/axes/text on the frame.")
    ap.add_argument("--size", type=float, default=MARKER_SIZE_METERS, help="Marker edge length in meters.")
    ap.add_argument("--dict", default=ARUCO_DICT_NAME, help="ArUco dictionary name.")
    ap.add_argument("--yaml", default=CALIBRATION_YAML, help="Calibration YAML path.")
    ap.add_argument("--res", default=f"{CAM_RESOLUTION[0]}x{CAM_RESOLUTION[1]}", help="WxH resolution, e.g. 1280x720")
    args = ap.parse_args()

    W,H = map(int, args.res.lower().split("x"))

    # Load calibration
    K, D = load_calibration(args.yaml)
    print("Loaded calibration:\n", K, "\n", D.ravel())

    # ArUco setup
    aruco_dict = get_aruco_dict(args.dict)
    detector, use_new_api = make_detector(aruco_dict)

    # Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (W, H)})
    picam2.configure(config)
    picam2.start(); time.sleep(0.2)

    use_gui = (not args.headless) and (os.getenv("DISPLAY") not in (None, "", "none"))
    if use_gui:
        try:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        except Exception:
            use_gui = False

    print("Press 'q' to quit." if use_gui else "Headless mode: Ctrl+C to stop.")
    last_print = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect
            try:
                corners, ids, _rej = detect(detector, use_new_api, gray)
            except Exception as e:
                # If aruco is truly broken, bail cleanly with a hint
                print(f"[ERROR] cv2.aruco crashed: {e}\n"
                      f"Tip: use a venv and `pip install opencv-contrib-python==4.4.0.46`")
                break

            if ids is not None and len(ids) > 0:
                # Pose (marker wrt camera)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, args.size, K, D)
                rvec = rvecs[0].reshape(3,1)
                tvec = tvecs[0].reshape(3,1)

                # Convert to camera wrt marker
                R_mc, _ = cv2.Rodrigues(rvec)
                R_cm = R_mc.T
                t_cm = (-R_cm @ tvec).reshape(3)
                yaw,pitch,roll = to_euler_zyx(R_cm)
                ypr_deg = tuple(map(math.degrees, (yaw,pitch,roll)))

                # Print at ~5 Hz in headless mode
                now = time.time()
                if args.headless and (now - last_print) > 0.2:
                    print(f"ID {int(ids[0])} | pos(m) {t_cm[0]:+.3f} {t_cm[1]:+.3f} {t_cm[2]:+.3f} | "
                          f"ypr(deg) {ypr_deg[0]:+.1f} {ypr_deg[1]:+.1f} {ypr_deg[2]:+.1f}")
                    last_print = now

                # Draw (safe path, no drawFrameAxes)
                if use_gui and not args.no_overlay:
                    try:
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    except Exception:
                        pass  # some builds glitch here; it's just decoration

                    # Manual axes via projectPoints (stable)
                    try:
                        draw_axes_manual(frame, K, D, rvec, tvec, args.size * 0.6)
                    except Exception:
                        pass

                    # Text overlay
                    y0 = 28; dy = 22
                    for i, line in enumerate(format_pose_text(t_cm, ypr_deg)):
                        cv2.putText(frame, line, (10, y0 + i*dy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,230,20), 2, cv2.LINE_AA)
            else:
                if use_gui and not args.no_overlay:
                    cv2.putText(frame, "No ArUco tag detected.", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Dict: {args.dict} | Tag size: {args.size*100:.0f} mm",
                                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

            if use_gui:
                if not safe_imshow(WINDOW_TITLE, frame):
                    # If GUI backend fails mid-run, drop to headless printing
                    use_gui = False
                else:
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break

    except KeyboardInterrupt:
        pass
    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass
        picam2.stop()

if __name__ == "__main__":
    main()
