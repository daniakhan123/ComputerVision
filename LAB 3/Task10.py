import numpy as np

s = 0.5
Scale = np.array([
    [s, 0, 0],
    [0, s, 0],
    [0, 0, 1]
])

theta = np.radians(20)
tx, ty = 100, 50

Rigid = np.array([
    [np.cos(theta), -np.sin(theta), tx],
    [np.sin(theta), np.cos(theta), ty],
    [0, 0, 1]
])

src = np.array([[0, 0], [100, 0], [0, 150], [100, 150]])
dst = np.array([[50, 40], [220, 20], [30, 300], [260, 280]])

A = []
for (x, y), (xp, yp) in zip(src, dst):
    A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y])
    A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y])

A = np.array(A)
B = dst.flatten()

h = np.linalg.solve(A, B)
Projective = np.append(h, 1).reshape(3, 3)

Final = Projective @ Rigid @ Scale

print(Final)

corner = np.array([0, 0, 1])
new_corner = Final @ corner
new_corner = new_corner / new_corner[2]
print(new_corner)
