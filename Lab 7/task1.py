import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)


def load_image(path="image.jpg"):
    img = cv2.imread(path)
    if img is None:
        print("image.jpg not found, using a sample image")
        img = np.full((300, 400, 3), 255, np.uint8)
        cv2.rectangle(img, (100, 130), (300, 270), (60, 120, 220), -1)
        cv2.fillPoly(img, [np.array([[80, 130], [200, 40], [320, 130]])], (40, 40, 180))
        cv2.rectangle(img, (180, 190), (220, 270), (30, 70, 30), -1)
        cv2.putText(img, "FAST", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def show(images, titles, name):
    plt.figure(figsize=(4 * len(images), 4))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/" + name + ".png", dpi=120)
    plt.show()


def part_a(img):
    h, w = img.shape[:2]
    move = np.float32([[1, 0, 80], [0, 1, 40]])
    translated = cv2.warpAffine(img, move, (w, h))
    turn = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
    rotated = cv2.warpAffine(img, turn, (w, h))
    scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
    shear = np.float32([[1, 0.4, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, shear, (int(w + 0.4 * h), h))
    show([img, translated, rotated, scaled, sheared],
         ["Original", "Translation", "Rotation", "Scaling", "Shear"], "q1a_2d")


def rot_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def part_b():
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(cube[i] != cube[j]) == 1]
    R = rot_z(np.radians(40)) @ rot_y(np.radians(30)) @ rot_x(np.radians(20))
    moved = cube @ R.T + np.array([3, 2, 1])

    ax = plt.figure(figsize=(7, 6)).add_subplot(111, projection="3d")
    for points, color in [(cube, "blue"), (moved, "red")]:
        for i, j in edges:
            ax.plot(*zip(points[i], points[j]), color=color)
    ax.plot([], [], color="blue", label="Original")
    ax.plot([], [], color="red", label="Rotated and moved")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.savefig("results/q1b_3d.png", dpi=120)
    plt.show()


def part_c():
    x, y = 3, 2
    p = np.array([x, y, 1])
    print("2D point:", (x, y))
    print("Homogeneous point:", p)
    T = np.array([[1, 0, 5], [0, 1, 4], [0, 0, 1]])
    print("After translating by (5, 4):", T @ p)
    print("""
Why the extra coordinate is used:
1. A 2x2 matrix cannot do translation. With the extra 1, translation becomes a matrix multiplication.
2. Rotation, scaling and translation can be combined by multiplying 3x3 matrices.
3. Perspective transforms need it. The last coordinate w is divided out (x/w, y/w), which makes far things look smaller.
4. Points at infinity can be written with w = 0.
""")


def part_d():
    size = 300
    grid = np.full((size, size, 3), 255, np.uint8)
    for k in range(0, size + 1, 30):
        cv2.line(grid, (k, 0), (k, size), (0, 0, 0), 1)
        cv2.line(grid, (0, k), (size, k), (0, 0, 0), 1)
    cv2.rectangle(grid, (0, 0), (size - 1, size - 1), (0, 0, 255), 2)

    before = np.float32([[0, 0], [size - 1, 0], [0, size - 1]])
    after = np.float32([[40, 60], [size - 60, 20], [80, size - 20]])
    M = cv2.getAffineTransform(before, after)
    print("Affine matrix:\n", M)
    result = cv2.warpAffine(grid, M, (size, size), borderValue=(255, 255, 255))
    show([grid, result], ["Original grid", "Affine transformed"], "q1d_affine")

    line = np.array([[0, 0], [100, 50], [200, 100]], float).reshape(-1, 1, 2)
    new = cv2.transform(line, M).reshape(-1, 2)
    a = new[1] - new[0]
    b = new[2] - new[0]
    print("Cross product of the transformed line (0 means still straight):", a[0] * b[1] - a[1] * b[0])
    print("Straight lines stay straight and parallel lines stay parallel, but angles and lengths can change.")


def part_e(img):
    h, w = img.shape[:2]
    flat = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    tilted_corners = np.float32([[60, 40], [w - 40, 90], [w - 120, h - 30], [20, h - 70]])
    tilted = cv2.warpPerspective(img, cv2.getPerspectiveTransform(flat, tilted_corners), (w, h),
                                 borderValue=(200, 200, 200))
    H = cv2.getPerspectiveTransform(tilted_corners, flat)
    fixed = cv2.warpPerspective(tilted, H, (w, h))
    print("Homography matrix:\n", np.round(H, 4))
    show([img, tilted, fixed], ["Original", "Tilted", "Corrected"], "q1e_homography")


image = load_image()
part_a(image)
part_b()
part_c()
part_d()
part_e(image)
