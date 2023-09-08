import cv2
import numpy as np
import random
from .hybrid_video import get_flow_from_images, image_transform_optical_flow

def get_prev_img_flow(flow_method, images, flow_factor, morpho_op, morpho_kernel, morpho_iter, idx, tween_steps, raft_model):
    ''' Prev img flow - uses image history to determine flow from 2nd to last image to prev_image
        Gets last stored image (nth2) and pass through or reshape, then get flow from new nth2 to prev_img, return prev_img
    '''
    # re-shape the nth2 image
    if morpho_op == "None" or morpho_iter == 0:
        # if no morphological operation or 0 iterations, pass through the nth2 image for flow
        shaped = images.get_nth(2)
    else:
        # else shape the image using morphological transformation
        shaped = morphological(images.get_nth(2), op=morpho_op, kernel=morpho_kernel, iterations=morpho_iter)

    # get flow from shaped nth2 image to 
    flow = get_flow_from_images(shaped, images.get_nth(1), flow_method, raft_model)
    print(f"Applying prev img flow to frame {idx} using {flow_method} flow from frames {idx-tween_steps-1} to {idx-1} at flow factor: {flow_factor}")

    return image_transform_optical_flow(images.get_nth(1), flow, flow_factor)

def morphological(img, op='open', kernel='rect', iterations=1, border=cv2.BORDER_DEFAULT):
    ''' performs morphological transformation with cv2.morphologyEx() '''

    # convert iterations to rounded int
    iterations = int(round(iterations))

    # convert keys to lowercase
    op.lower()
    kernel.lower()

    # convert to grayscale to avoid each channel being treated separately by 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # handles to cv2 constants dicts
    structurings = {
        'rect': cv2.MORPH_RECT,
        'cross': cv2.MORPH_CROSS,
        'ellipse': cv2.MORPH_ELLIPSE
    }
    operations = {
        'dilate': cv2.MORPH_DILATE,
        'erode': cv2.MORPH_ERODE,
        'open': cv2.MORPH_OPEN,         # same as dilate(erode(src,element))
        'close': cv2.MORPH_CLOSE,       # same as erode(dilate(src,element))
        'gradient': cv2.MORPH_GRADIENT, # same as dilate(src,element) − erode(src,element)
        'tophat': cv2.MORPH_TOPHAT,     # same as src − open(src,element)
        'blackhat': cv2.MORPH_BLACKHAT  # same as close(src,element) − src
    }

    # get structuring element from dict
    kernel_code = cv2.getStructuringElement(structurings.get(kernel), (5, 5))

    # get morphological operation from dict
    if op == 'random': # random operation selected
        operation_code = random.choice(list(operations.keys()))
    else:
        operation_code = operations.get(op)

    # convert back to bgr
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # console reporting
    print(f"Prev img flow morphological kernel:{kernel} operation:{op} for {iterations} iteration{'s' if iterations > 1 else ''}")

    # use morphological operation
    return cv2.morphologyEx(src=img, op=operation_code, kernel=kernel_code, iterations=iterations, borderType=border)

'''
Composition of the structuring (kernel) element:

cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)

# Elliptical Kernel
cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)

# Cross-shaped Kernel
cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
'''