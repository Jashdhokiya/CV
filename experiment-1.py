# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV Version:", cv2.__version__)
print("NumPy Version :", np.__version__)

# %%
# Read Image
img = cv2.imread("Data/shoes.png")

if img is None:
    print("Image not found!")
else:
    print("Image loaded successfully.")
    print("Shape :", img.shape)
    print("Data Type :", img.dtype)

# %%
# Display Image (OpenCV)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Display Image (Matplotlib)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8,6))
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")
plt.show()

# %%
# Details of Image
height, width, channels = img.shape

print("Height :", height)
print("Width  :", width)
print("Channels :", channels)

# %%
# Access Pixel
print("Pixel Value at (100,100):")
print(img[100,100])

# %%
#Modify Pixel
modified = img.copy()

modified[100,100] = [0,255,0]

cv2.imshow("Modified Pixel", modified)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Region of Interest (ROI/Crop)
roi = img[100:300,100:300]

cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Resize
resize = cv2.resize(img,(400,400))

cv2.imshow("Resized", resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Flip
horizontal = cv2.flip(img,1)
vertical = cv2.flip(img,0)

cv2.imshow("Horizontal Flip",horizontal)
cv2.imshow("Vertical Flip",vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Rotate
rotate = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Rotated",rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Draw Rectangle 
rectangle = img.copy()

cv2.rectangle(rectangle,
              (100,100),
              (400,300),
              (0,255,0),
              3)

cv2.imshow("Rectangle",rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Add text
text = img.copy()

cv2.putText(text,
            "Computer Vision Lab",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2)

cv2.imshow("Text",text)
cv2.waitKey(0)
cv2.destroyAllWindows()

