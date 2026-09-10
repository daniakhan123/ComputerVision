import numpy as np

src = np.array([[0, 0], [1, 0], [0, 1]])
dst = np.array([[1, 2], [4, 3], [2, 6]])

A = []
B = []

for (x, y), (xp, yp) in zip(src, dst):
    A.append([x, y, 1, 0, 0, 0])
    A.append([0, 0, 0, x, y, 1])
    B.append(xp)
    B.append(yp)

A = np.array(A)
B = np.array(B)

params = np.linalg.solve(A, B)
a, b, c, d, e, f = params

Affine = np.array([
    [a, b, c],
    [d, e, f],
    [0, 0, 1]
])

print(Affine)
