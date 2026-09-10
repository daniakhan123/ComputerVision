import numpy as np

k = -0.5

shear_matrix = np.array([
    [1, k],
    [0, 1]
])

print("Shear Matrix:")
print(shear_matrix)

x = 100
y = 50

new_x = x + k * y
new_y = y

print("Original Pixel:", (x, y))
print("New Pixel:", (new_x, new_y))
