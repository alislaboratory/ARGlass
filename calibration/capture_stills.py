#!/usr/bin/env python3
import os
import cv2
from picamera2 import Picamera2
import time
import glob

# ??? Configuration ???
SAVE_DIR = "stills"
PREVIEW_SIZE = (640, 480)   # adjust as needed

# ??? Ensure save directory exists ???
os.makedirs(SAVE_DIR, exist_ok=True)

# ??? Initialize Picamera2 ???
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": PREVIEW_SIZE}
)
picam2.configure(preview_config)
picam2.start()
time.sleep(2)  # camera warm-up

# ??? Determine starting index for filenames ???
existing = glob.glob(os.path.join(SAVE_DIR, "*.jpg"))
next_idx = len(existing) + 1

print("Preview started. Press Enter to capture, Ctrl+C to quit.")

try:
    while True:
        # Capture frame and flip 180°
        frame = picam2.capture_array()
        flipped = cv2.rotate(frame, cv2.ROTATE_180)

        # Show preview
        cv2.imshow("Preview (flipped)", flipped)
        key = cv2.waitKey(1)

        # On Enter (CR or LF), save
        if key in (10, 13):
            filename = os.path.join(SAVE_DIR, f"img_{next_idx:03d}.jpg")
            cv2.imwrite(filename, flipped)
            print(f"Saved {filename}")
            next_idx += 1

except KeyboardInterrupt:
    pass

# ??? Cleanup ???
cv2.destroyAllWindows()
picam2.stop()
print("\nExiting.")
