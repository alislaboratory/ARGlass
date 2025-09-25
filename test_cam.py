import cv2
for idx in range(0,3):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    print(idx, "opened:", cap.isOpened())
    ok, frame = cap.read()
    print(idx, "read frame:", ok, "shape:", getattr(frame, "shape", None))
    cap.release()