
import cv2
import numpy as np
from helpers import show

img = cv2.imread("images/car.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_tight = np.array([24, 230, 230])
upper_tight = np.array([27, 255, 255])
tight_mask = cv2.inRange(hsv, lower_tight, upper_tight)

lower_final = np.array([15, 100, 80])
upper_final = np.array([35, 255, 255])
final_mask = cv2.inRange(hsv, lower_final, upper_final)

tight_result = cv2.bitwise_and(img, img, mask=tight_mask)
final_result = cv2.bitwise_and(img, img, mask=final_mask)

show([img, tight_mask, final_mask, tight_result, final_result],
     ["Original", "Too restrictive mask", "Final mask", "Extracted (restrictive)", "Extracted (final)"],
     "task4")
cv2.imwrite("results/task4_final_mask.png", final_mask)
cv2.imwrite("results/task4_final_extracted.png", final_result)

print("Restrictive range lower", lower_tight, "upper", upper_tight, "-> pixels:", cv2.countNonZero(tight_mask))
print("Final range       lower", lower_final, "upper", upper_final, "-> pixels:", cv2.countNonZero(final_mask))
print()
print("Why the restrictive mask fails")
print("It only accepts a very narrow hue and requires very high saturation and brightness.")
print("The car has shaded and highlighted parts that are darker or paler yellow, so those pixels fall outside the range and the mask has holes.")
print("The final range keeps the yellow hue but allows lower saturation and brightness, so the whole car is captured.")
