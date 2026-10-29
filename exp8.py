import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Create image with 3 known lines
img = np.zeros((400, 600), dtype=np.uint8)

known_lines = [(0.5, 40), (-0.2, 200), (0.4, 150)]

for m, c in known_lines:
    x1, x2 = 0, 599
    y1 = int(m * x1 + c)
    y2 = int(m * x2 + c)

    cv2.line(img, (x1, y1), (x2, y2), 255, 2)

# Display simple image with lines
plt.figure(figsize=(7, 4))
plt.imshow(img, cmap="gray")
plt.title("Image with Three Lines")
plt.axis("off")
plt.show()

# 2. Canny Edge Detection
edges = cv2.Canny(img, 50, 150)
y, x = np.where(edges > 0)

# 3. Line Detection using Hough Transform
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=10
)

result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(7, 4))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines using Hough Transform")
plt.axis("off")
plt.show()

# 4. Hough Transform in (m, c) Space
m_values = np.linspace(-1.2, 1.2, 300)

plt.figure(figsize=(7, 5))

# Each image point gives c = y - mx
for i in range(0, len(x), 20):
    c_values = y[i] - m_values * x[i]
    plt.plot(m_values, c_values, color="gray", alpha=0.2)

for i, (m, c) in enumerate(known_lines, 1):
    plt.scatter(m, c, color="red", label=f"Line {i}")

plt.title("(m, c) Parameter Space")
plt.xlabel("Slope (m)")
plt.ylabel("Intercept (c)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 5. Hough Transform in (r, theta) Space
theta_deg = np.arange(180)
theta = np.deg2rad(theta_deg)

r_max = int(np.ceil(np.hypot(600, 400)))
accumulator = np.zeros((2 * r_max + 1, 180), dtype=np.uint16)

for px, py in zip(x, y):
    r = np.rint(px * np.cos(theta) + py * np.sin(theta)).astype(int)
    accumulator[r + r_max, np.arange(180)] += 1

# Find strong accumulator peaks
smooth = cv2.GaussianBlur(
    accumulator.astype(np.float32), (5, 5), 0
)

temp = smooth.copy()
peaks = []

for _ in range(3):
    ri, ti = np.unravel_index(np.argmax(temp), temp.shape)
    peaks.append((ri, ti))

    # Suppress nearby peaks to avoid selecting the same peak repeatedly
    temp[max(0, ri-5):min(temp.shape[0], ri+6),
         max(0, ti-5):min(temp.shape[1], ti+6)] = 0

# Plot sinusoidal curves and mark peaks
plt.figure(figsize=(8, 5))

for i in range(0, len(x), 20):
    r = x[i] * np.cos(theta) + y[i] * np.sin(theta)
    plt.plot(theta_deg, r, color="gray", alpha=0.15)

for i, (ri, ti) in enumerate(peaks, 1):
    rho = ri - r_max
    plt.scatter(
        ti, rho, color="red", marker="x", s=100,
        label=f"Peak {i}: r={rho}, theta={ti}°"
    )
    print(f"Peak {i}: r={rho}, theta={ti} degrees")

plt.title("(r, theta) Hough Space")
plt.xlabel("Theta (degrees)")
plt.ylabel("Rho (r)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()