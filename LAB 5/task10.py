
import cv2
import numpy as np
from helpers import show

gray = cv2.imread("images/document.png", 0)

_, global_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
otsu_value, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

methods = [("Global T=100", global_mask), ("Adaptive Gaussian 31, C=15", adaptive_mask), ("Otsu T=" + str(int(otsu_value)), otsu_mask)]
show([gray] + [m for n, m in methods], ["Original"] + [n for n, m in methods], "task10")

part = gray.shape[1] // 3
print("Black pixels (%) in the left third and right third of the page (text is spread evenly, so they should match)")
print("Method                      Left    Right   Difference")
scores = []
for name, mask in methods:
    left = 100 * np.mean(mask[:, :part] == 0)
    right = 100 * np.mean(mask[:, -part:] == 0)
    scores.append(abs(left - right))
    print(name.ljust(27), str(round(left, 1)).ljust(7), str(round(right, 1)).ljust(7), round(abs(left - right), 1))

best = scores.index(min(scores))
cv2.imwrite("results/task10_best_mask.png", methods[best][1])

print()
print("Global thresholding")
print("1. Assumes: the lighting is the same everywhere, so one number separates text from paper.")
print("2. Identifies: the text on the bright side of the page.")
print("3. Wrong: the dark side, where the paper itself becomes black.")
print("4. Most important parameter: the threshold value.")
print("Adaptive thresholding")
print("1. Assumes: lighting changes slowly, so text is darker than its local neighbourhood.")
print("2. Identifies: the text on the whole page, bright and dark sides.")
print("3. Wrong: it can create small noise specks in flat areas if C is too small.")
print("4. Most important parameter: C (and the block size).")
print("Otsu thresholding")
print("1. Assumes: the histogram has two groups (text and paper) with one global split.")
print("2. Identifies: the bright side of the page, where text and paper are clearly separated.")
print("3. Wrong: the dark side, because the histogram mixes dark paper with text.")
print("4. Most important parameter: none to tune, but it depends on the lighting in the image.")
print()
print("Conclusion")
print("Most meaningful result (left and right ink amounts closest):", methods[best][0])
print("Why: the page has uneven light, and only a method that works on local neighbourhoods is not fooled by the brightness change.")
