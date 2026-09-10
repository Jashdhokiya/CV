def median_filter(image, kernel_size = 3):
    image = image.copy()
    image = cv2.copyMakeBorder(image, int(kernel_size / 2), int(kernel_size / 2), int(kernel_size / 2), int(kernel_size / 2), cv2.BORDER_REPLICATE)
    print(image.shape)
    currentImage = image[0:kernel_size, 0:kernel_size]
    for i in range(image.shape[0] - kernel_size):
        for j in range(image.shape[1] - kernel_size):
            currentImage = image[i:i + kernel_size, j:j + kernel_size]
            arr1 = currentImage.flatten()  
            arr1.sort();
            median = arr1[int(len(arr1) / 2)];
            image[i + int(kernel_size / 2), j + int(kernel_size / 2)] = median
    
    return image;

filterImage = median_filter(noisy, 3)
plt.imshow(filterImage, cmap="gray")

def mean_filter(image, kernel_size = 3):
    image = image.copy()
    image = cv2.copyMakeBorder(image, int(kernel_size / 2), int(kernel_size / 2), int(kernel_size / 2), int(kernel_size / 2), cv2.BORDER_REPLICATE)

    currentImage = image[0:kernel_size, 0:kernel_size]
    for i in range(image.shape[0] - kernel_size):
        for j in range(image.shape[1] - kernel_size):
            currentImage = image[i:i + kernel_size, j:j + kernel_size]
            mean = np.mean(currentImage)
            image[i + int(kernel_size / 2), j + int(kernel_size / 2)] = mean
    return image;

filterMean = mean_filter(noisy, 3)
plt.imshow(filterMean, cmap="gray")