
import cv2
import numpy as np
from helpers import show

img = cv2.imread("images/coins.png")
true_count = 9


def watershed_with(fraction):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fraction * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    count, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    result = img.copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = (0, 0, 255)
    regions = len(np.unique(markers[markers > 1]))
    return count - 1, regions, result


fractions = [0.2, 0.4, 0.7, 0.9]
images = []
titles = []
good = []
print("Experiment  Distance Threshold  Foreground  Regions  Observed Result")
for i, fraction in enumerate(fractions):
    foreground, regions, result = watershed_with(fraction)
    if regions < true_count:
        observed = "coins still merged"
    elif regions > true_count:
        observed = "coins split into extra pieces"
    else:
        observed = "all coins separated correctly"
        good.append(fraction)
    print(str(i + 1).ljust(11), str(fraction).ljust(19), str(foreground).ljust(11), str(regions).ljust(8), observed)
    images.append(result)
    titles.append("Threshold " + str(fraction) + " x max: " + str(regions) + " regions")

show(images, titles, "task8", 4)
cv2.imwrite("results/task8_best_result.png", watershed_with(good[0] if good else 0.7)[2])

print()
print("True number of coins in the image:", true_count)
if good:
    print("Most meaningful separation: distance threshold between", min(good), "and", max(good), "of the maximum distance")
else:
    print("No experiment matched the true count. Try values between the tested ones.")
print("Too low: sure foreground is too big, touching coins share one marker and stay merged.")
print("Too high: sure foreground shrinks to tiny cores. Here 0.9 still worked, but values close to 1 risk losing small coins or adding extra markers on noisy ones.")
