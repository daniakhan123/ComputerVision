
import cv2
import numpy as np
from helpers import show

img = cv2.imread("images/part.png", 0)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
strength = np.abs(gx) + np.abs(gy)

pairs = [(10, 30), (40, 100), (40, 200), (40, 500)]
results = []
for low, high in pairs:
    results.append(cv2.Canny(blurred, low, high))

show([img] + results,
     ["Original"] + ["Edge Result " + str(i + 1) + " (low=" + str(p[0]) + ", high=" + str(p[1]) + ")" for i, p in enumerate(pairs)],
     "task5")
cv2.imwrite("results/task5_best_edges.png", results[1])

print("Low  High  Strong pixels  Weak pixels  Final edge pixels")
for (low, high), edges in zip(pairs, results):
    strong = int(np.sum(strength > high))
    weak = int(np.sum((strength > low) & (strength <= high)))
    print(str(low).ljust(4), str(high).ljust(5), str(strong).ljust(14), str(weak).ljust(12), cv2.countNonZero(edges))

print()
print("Strong edges: the outline of the part and the hole. Their gradient is far above the high threshold, so they are kept.")
print("Weak edges: the faint scratch at the bottom and the small low-contrast box. Their gradient is about 90 to 115, so they survive only when high is 100 or lower.")
print("Missing edges: with high=500 even the part outline breaks, and with high=200 the weak scratch and small box are gone.")
print("Unwanted edges: with low=10 and high=30 the image noise and the background texture create many small edges that are not boundaries.")
print()
print("How the thresholds change the result")
print("Hysteresis keeps pixels above high, drops pixels below low, and keeps in-between pixels only when they touch a strong edge.")
print("Raising high (low fixed) removes weak edges and noise but can miss real faint boundaries. Lowering low lets more weak edges join and adds noise.")
print("Best choice here: low=40, high=100 keeps the outline and the faint real edges without the noise.")
