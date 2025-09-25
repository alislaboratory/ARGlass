#!/usr/bin/env python3
import cv2
import time
from picamera2 import Picamera2

# ?? Configuration ??
CALIB_FILE    = "camera_calibration.yaml"
PREVIEW_SIZE  = (640, 480)
FLIP_180      = True    # set False if camera is upright
WINDOW_NAME   = "Live Undistorted Preview (q to quit)"

# ?? Load calibration ??
fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise IOError(f"Cannot open calibration file: {CALIB_FILE}")
cam_mtx = fs.getNode("camera_matrix").mat()
dist    = fs.getNode("distortion_coefficients").mat()
fs.release()

# ?? Precompute undistort map ??
new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
    cam_mtx, dist, PREVIEW_SIZE, 1, PREVIEW_SIZE
)
map1, map2 = cv2.initUndistortRectifyMap(
    cam_mtx, dist, None, new_cam_mtx, PREVIEW_SIZE, cv2.CV_16SC2
)

# ?? Start camera ??
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format":"RGB888", "size":PREVIEW_SIZE}
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # warm-up

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
print("Press 'q' to exit.")

try:
    while True:
        frame = picam2.capture_array()
        # undistort
        undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        # flip if still upside-down
        if FLIP_180:
            undist = cv2.rotate(undist, cv2.ROTATE_180)
        cv2.imshow(WINDOW_NAME, undist)

        # exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
