
import cv2
import numpy as np
from helpers import show

img = cv2.imread("images/mri.png", 0)


def region_grow(image, seed, threshold):
    height, width = image.shape
    mask = np.zeros((height, width), np.uint8)
    seed_value = int(image[seed[1], seed[0]])
    stack = [seed]
    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if mask[y, x] == 255:
            continue
        if abs(int(image[y, x]) - seed_value) > threshold:
            continue
        mask[y, x] = 255
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    return mask


seeds = [(95, 90), (150, 70)]
thresholds = [10, 25, 50]

marked = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x, y in seeds:
    cv2.circle(marked, (x, y), 5, (0, 0, 255), -1)

images = [marked]
titles = ["Original (red dots = seeds)"]
print("Seed        Threshold  Region pixels")
for seed in seeds:
    for t in thresholds:
        mask = region_grow(img, seed, t)
        images.append(mask)
        titles.append("Seed " + str(seed) + " threshold " + str(t))
        print(str(seed).ljust(11), str(t).ljust(10), cv2.countNonZero(mask))

show(images, titles, "task6", 4)
cv2.imwrite("results/task6_final_mask.png", region_grow(img, seeds[0], 25))

print()
print("Parameters: 4-connected neighbours, thresholds 10, 25 and 50, seeds", seeds)
print("Threshold 10 is strict, 25 is moderate and 50 is loose enough to leak into other tissue.")
print()
print("Why can changing the seed point significantly change the final segmented region?")
print("Every pixel is compared with the intensity of the seed, and the region can only grow through connected neighbours.")
print("A seed in another tissue has a different intensity, so the accepted range changes and the region grows somewhere else.")
print("Even a noisy seed pixel shifts the accepted range a little, which is why the seed should be chosen carefully.")
