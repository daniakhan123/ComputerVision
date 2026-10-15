
import cv2
import numpy as np
from helpers import show

img = cv2.imread("images/coins.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

fraction = 0.7
_, sure_fg = cv2.threshold(dist, fraction * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

count, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

result = img.copy()
markers = cv2.watershed(result, markers)
result[markers == -1] = (0, 0, 255)

dist_view = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show([img, thresh, sure_bg, dist_view, sure_fg, unknown, result],
     ["Original", "Threshold", "Sure Background", "Distance Transform", "Sure Foreground", "Unknown Region", "Final Watershed"],
     "task7", 4)
cv2.imwrite("results/task7_watershed.png", result)

print("Sure foreground markers (coins found):", count - 1)
print("Parameters: blur 5x5, Otsu threshold, opening 3x3 x2, dilation x3, distance threshold =", fraction, "x max distance")
print("Blur removes noise, Otsu picks the threshold automatically, opening removes small specks,")
print("and 0.7 of the max distance keeps only the centre of each coin so touching coins get separate markers.")
print("Look at the red lines in the final image to check that touching coins were separated.")
