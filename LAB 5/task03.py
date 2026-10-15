
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/coins.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print("Otsu threshold on the original image:", t)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.hist(gray.ravel(), bins=256, range=[0, 256])
plt.axvline(t, color="red")
plt.title("Histogram (red line = Otsu threshold)")
plt.subplot(1, 3, 3)
plt.imshow(otsu, cmap="gray")
plt.title("Otsu binary mask")
plt.axis("off")
plt.tight_layout()
plt.savefig("results/task3.png", dpi=110)
plt.show()
cv2.imwrite("results/task3_otsu_mask.png", otsu)

contrast = cv2.convertScaleAbs(gray, alpha=1.4, beta=-40)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
darker = cv2.convertScaleAbs(gray, alpha=1.0, beta=-40)
print()
print("Version              Otsu threshold")
print("Original            ", t)
for name, version in [("More contrast", contrast), ("Blurred", blurred), ("Darker", darker)]:
    value, mask = cv2.threshold(version, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(name.ljust(20), value)

print()
print("Why is Otsu useful when the programmer does not know the correct threshold beforehand?")
print("Otsu reads the histogram and picks the value that best separates the two groups (object and background).")
print("When the lighting or contrast changes, the threshold moves with it, so nobody has to tune it by hand for every image.")
