import cv2, glob, os

CHECKERBOARD = (8, 5)  # adjust to your board
flags        = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

for fname in glob.glob("/stills/*.jpg"):
    img = cv2.imread(fname)
    if img is None:
        print("x", fname, "→ load failed")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    print(os.path.basename(fname), "→ found?", ret)
    disp = img.copy()
    if ret:
        cv2.drawChessboardCorners(disp, CHECKERBOARD, corners, ret)
    cv2.imshow("debug", disp)
    cv2.waitKey(0)

cv2.destroyAllWindows()
