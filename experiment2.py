import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

image = cv2.imread("data/shoes.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap="gray")

def add_salt_pepper_noise(image, probability = 0.3):
    noisy = image.copy()

    random = np.random.rand(*image.shape)

    noisy[random < probability / 2] = 0
    noisy[random > 1 - probability / 2] = 255

    return noisy

prob = 0.4

noisy = add_salt_pepper_noise(image, prob) 

plt.imshow(noisy, cmap="gray")