import numpy as np

theta = np.radians(45)
s = 2
tx, ty = 4, 6

S = np.array([
    [s, 0, 0],
    [0, s, 0],
    [0, 0, 1]
])

R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

T = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
])

Similarity = T @ R @ S

point = np.array([1, 1, 1])
new_point = Similarity @ point

print(Similarity)
print(new_point)
