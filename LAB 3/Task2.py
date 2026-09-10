import numpy as np

angle = -45

theta = np.radians(angle)

cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

rotation_matrix = np.array([
    [cos_theta, -sin_theta],
    [sin_theta, cos_theta]
])

print("Rotation Matrix:")
print(rotation_matrix)

# Original dimensions
W = 400
H = 300

new_W = abs(W * cos_theta) + abs(H * sin_theta)
new_H = abs(W * sin_theta) + abs(H * cos_theta)

print("New Width:", int(np.ceil(new_W)))
print("New Height:", int(np.ceil(new_H)))
