import numpy as np

src = np.array([[50, 300], [250, 300], [50, 100], [250, 100]])
dst = np.array([[0, 300], [300, 300], [0, 0], [300, 0]])

A = []

for (x, y), (xp, yp) in zip(src, dst):
    A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y])
    A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y])

A = np.array(A)
B = dst.flatten()

h = np.linalg.solve(A, B)
H = np.append(h, 1).reshape(3, 3)

print(H)
