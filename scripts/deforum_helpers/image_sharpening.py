import cv2
import numpy as np

def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0, mask=None):
    if amount == 0:
        return img
    # Return a sharpened version of the image, using an unsharp mask.
    # If mask is not None, only areas under mask are handled
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    if mask is not None:
        mask = np.array(mask)
        masked_sharpened = cv2.bitwise_and(sharpened, sharpened, mask=mask)
        masked_img = cv2.bitwise_and(img, img, mask=255-mask)
        sharpened = cv2.add(masked_img, masked_sharpened)
    return sharpened