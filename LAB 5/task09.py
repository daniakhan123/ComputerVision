
import cv2
import numpy as np
from helpers import show

img = cv2.imread("images/nature.png")
pixels = img.reshape((-1, 3))
pixels = np.float32(pixels)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
cv2.setRNGSeed(0)


def colour_name(bgr):
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 180 else "grey"
    if h < 10 or h >= 170:
        return "red"
    if h < 22:
        return "brown" if v < 150 else "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple"


original_colours = len(np.unique(img.reshape(-1, 3), axis=0))
images = [img]
titles = ["Original"]
for k in (2, 4, 6):
    ret, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    images.append(segmented)
    titles.append("K=" + str(k))
    cv2.imwrite("results/task9_k" + str(k) + ".png", segmented)

    difference = np.mean(np.abs(img.astype(int) - segmented.astype(int)))
    print("K =", k)
    print("  Number of clusters:", k)
    print("  Average difference from original (0-255):", round(difference, 1))
    print("  Colours: original", original_colours, "-> segmented", k)
    for c in range(k):
        share = 100 * np.mean(labels.flatten() == c)
        print("  Cluster", c, colour_name(centers[c]), "BGR", centers[c], str(round(share, 1)) + "% of pixels")

show(images, titles, "task9")
print()
print("Small K gives strong simplification but merges different regions. Large K keeps more detail and looks closer to the original.")
