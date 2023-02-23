import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

def frst(img, radii, alpha):
    #resize the image to workin_width, auto scale height
    working_width = 128
    working_height = int(img.shape[0] * working_width / img.shape[1])
    img = cv2.resize(img, (working_width, working_height))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient and Laplacian of the image
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    # Initialize the output image
    output = np.zeros(img.shape[:2], dtype=np.float32)

    # Compute the FRST at each radius and orientation
    for r in radii:
        for theta in range(0, 360, alpha):
            c, s = np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)
            x, y = np.meshgrid(range(-r, r+1), range(-r, r+1))
            mask = np.logical_or(np.abs(x*c + y*s) < r/2, np.abs(x*s - y*c) < r/2)
            filtered = np.abs(cv2.filter2D(laplacian, cv2.CV_32F, mask.astype(np.float32))) / np.sum(np.abs(mask))
            output = np.maximum(output, filtered)

    # Normalize the output image
    output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return output

images =[]
for i in range(1, 6):
    prev_img_cv2 = cv2.imread("imgs/" + str(i) + ".png")
    radii = [1]
    alpha = 15

    # Compute the FRST
    frst_img = frst(prev_img_cv2, radii, alpha)
    #add image to images
    images.append(frst_img)

#show all images in single plot
#set figsize to sqrt of number of images, rounded up
figsize = int(np.ceil(np.sqrt(len(images))))
fig=plt.figure(figsize=(figsize, figsize))
for i in range(1, len(images)+1):
    fig.add_subplot(figsize, figsize, i)
    plt.imshow(images[i-1], cmap='gray')
plt.show()