import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_perspective_transform(img, src_pts, dst_pts, out_size):
    H = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    return cv2.warpPerspective(img, H, out_size), H


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def pick_corners(img):
    points = []
    shown = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(shown, (x, y), 6, (0, 0, 255), -1)

    cv2.namedWindow("Click 4 corners")
    cv2.setMouseCallback("Click 4 corners", on_click)
    while len(points) < 4:
        cv2.imshow("Click 4 corners", shown)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    return points


def make_sample():
    doc = np.full((400, 300, 3), 255, np.uint8)
    cv2.rectangle(doc, (0, 0), (299, 399), (0, 0, 0), 4)
    for i, text in enumerate(["COMPUTER", "VISION", "LAB 07", "FAST NUCES"]):
        cv2.putText(doc, text, (30, 90 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 40, 20), 2)
    flat = np.float32([[0, 0], [299, 0], [299, 399], [0, 399]])
    tilt = np.float32([[120, 60], [430, 20], [480, 470], [60, 400]])
    H = cv2.getPerspectiveTransform(flat, tilt)
    return cv2.warpPerspective(doc, H, (560, 520), borderValue=(180, 180, 180)), tilt


if len(sys.argv) > 1:
    img = cv2.imread(sys.argv[1])
    corners = order_points(pick_corners(img))
else:
    img, corners = make_sample()
    corners = order_points(corners)

tl, tr, br, bl = corners
width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
rectangle = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

result, H = apply_perspective_transform(img, corners, rectangle, (width, height))
print("Homography matrix:\n", np.round(H, 4))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Tilted view")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Corrected view")
plt.axis("off")
plt.tight_layout()
plt.savefig("q2_result.png", dpi=120)
plt.show()
