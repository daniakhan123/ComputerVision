import numpy as np

src = np.array([[120, 80], [300, 90], [100, 250], [310, 260]])
dst = np.array([[400, 100], [580, 100], [390, 270], [590, 270]])

A = []

for (x, y), (xp, yp) in zip(src, dst):
    A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y])
    A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y])

A = np.array(A)
B = dst.flatten()

h = np.linalg.solve(A, B)
H = np.append(h, 1).reshape(3, 3)

print(H)

point = np.array([120, 80, 1])
new_point = H @ point
new_point = new_point / new_point[2]
print(new_point)
