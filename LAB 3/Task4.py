import numpy as np

tx = 150
ty = 80

translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
])

print("Translation Matrix:")
print(translation_matrix)

point = np.array([
    [100],
    [50],
    [1]
])

new_point = translation_matrix @ point

print("Original Point:")
print(point)

print("Translated Point:")
print(new_point)
