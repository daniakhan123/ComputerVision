
import cv2
import numpy as np
from helpers import show

gray = cv2.imread("images/document.png", 0)
methods = [("Mean", cv2.ADAPTIVE_THRESH_MEAN_C), ("Gaussian", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)]
rows = []


def count_specks(mask):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - mask)
    return int(np.sum(stats[1:, cv2.CC_STAT_AREA] <= 3))


images = [gray]
titles = ["Original"]
for name, method in methods:
    for block in (11, 31, 71):
        mask = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, block, 10)
        images.append(mask)
        titles.append(name + " block=" + str(block) + " C=10")
        rows.append((name, block, 10, 100 * np.mean(mask == 0), count_specks(mask)))
show(images, titles, "task2_block_sizes", 4)

images = [gray]
titles = ["Original"]
for name, method in methods:
    for c in (2, 10, 30):
        mask = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, 31, c)
        images.append(mask)
        titles.append(name + " block=31 C=" + str(c))
        rows.append((name, 31, c, 100 * np.mean(mask == 0), count_specks(mask)))
show(images, titles, "task2_c_values", 4)

print("Method    Block  C   Black%  Noise specks")
for name, block, c, black, specks in rows:
    print(name.ljust(9), str(block).ljust(6), str(c).ljust(3), str(round(black, 1)).ljust(7), specks)

middle = np.median([r[3] for r in rows])
good = [r for r in rows if abs(r[3] - middle) < 0.4 * middle]
best = min(good, key=lambda r: r[4])
cv2.imwrite("results/task2_best_mask.png", cv2.adaptiveThreshold(
    gray, 255, dict(methods)[best[0]], cv2.THRESH_BINARY, best[1], best[2]))

mean_noise = sum(r[4] for r in rows if r[0] == "Mean")
gauss_noise = sum(r[4] for r in rows if r[0] == "Gaussian")
better = "Gaussian" if gauss_noise <= mean_noise else "Mean"

print()
print("1. Neighbourhood too small: the threshold follows tiny local changes, so noise becomes black specks and the inside of thick strokes can turn hollow or broken.")
print("2. Neighbourhood too large: the threshold behaves like a global one, so the uneven lighting problem returns and faint text in dark areas can disappear.")
print("3. C increased: the threshold gets lower, so fewer pixels turn black. The background gets cleaner but thin strokes can break. A very small C gives noise.")
print("4. Cleanest combination on this image:", best[0], "block", best[1], "C", best[2], "(fewest noise specks with a normal amount of text)")
print("5. Better method on this image:", better, "(noise specks Mean =", mean_noise, ", Gaussian =", gauss_noise, ")")
