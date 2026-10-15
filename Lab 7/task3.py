import os
import sys
import glob
import cv2
import numpy as np

BOARD = (9, 6)
SQUARE = 1.0
FOLDER = "calib_images"
CALIB_FILE = "camera_calib.npz"


def capture():
    os.makedirs(FOLDER, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = len(glob.glob(FOLDER + "/*.jpg"))
    print("Press s to save a photo and q to quit. Take 15 to 20 photos from different angles.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD)
        preview = frame.copy()
        if found:
            cv2.drawChessboardCorners(preview, BOARD, corners, found)
        cv2.putText(preview, "saved: " + str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and found:
            cv2.imwrite(FOLDER + "/img_" + str(count).zfill(2) + ".jpg", frame)
            count += 1
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def calibrate():
    board_points = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    board_points[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE

    object_points = []
    image_points = []
    paths = glob.glob(FOLDER + "/*.jpg")
    if not paths:
        print("No images found. Run capture first.")
        return

    for path in paths:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            object_points.append(board_points)
            image_points.append(corners)

    print(len(image_points), "of", len(paths), "images had a chessboard")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    errors = []
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
        errors.append(cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(projected))
    print("Mean reprojection error:", np.mean(errors))

    np.savez(CALIB_FILE, K=K, dist=dist)
    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", dist.ravel())
    print("Saved to", CALIB_FILE)


def live():
    data = np.load(CALIB_FILE)
    K = data["K"]
    dist = data["dist"]

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    print("Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fixed = cv2.undistort(frame, K, dist, None, new_K)
        cv2.imshow("Original and Undistorted", np.hstack([frame, fixed]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


mode = sys.argv[1] if len(sys.argv) > 1 else "live"
if mode == "capture":
    capture()
elif mode == "calibrate":
    calibrate()
else:
    live()
