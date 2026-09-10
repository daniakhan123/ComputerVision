import numpy as np

theta = np.radians(30)
tx, ty = 5, 3

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

Rigid = T @ R

point = np.array([2, 1, 1])
new_point = Rigid @ point

print(Rigid)
print(new_point)
