from glob import glob
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import time

# The point in the image you want to align the symmetry to
# Maximum number of pixels arount target_location to search for symmetry
symmetry_max_offset = 20; 
def showImage(img):
    plt.imshow(img)
    plt.show()
def generate_sinusoidal_gradient(shape, center, frequency, amplitude):
    # Create a meshgrid of the x and y coordinates
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # Calculate the distances from the center of the image
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Calculate the phase of the sinusoidal gradient for each pixel
    phase = 2*np.pi*frequency*r
    # Calculate the values of the sinusoidal gradient for each pixel
    gradient = amplitude*np.sin(phase)
    # Scale the values to the range [0, 255]
    gradient = ((gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient))) * 255
    # Convert the gradient to an 8-bit integer array
    gradient = gradient.astype(np.uint8)
    return gradient
def generate_radial_lines(shape, center, num_slices):
  cx, cy = 200, 300    # (x,y) coordinates of circle centre
  N      = 16          # number of slices in our pie
  l      = cx      # length of radial lines - larger than necessary
  im = np.zeros((shape[1],shape[0]), np.uint8)

  # Create each sector in white on black background
  for sector in range(num_slices):
      if sector % 2 == 0:
        continue 
      startAngle = sector * 360/num_slices
      endAngle   = startAngle + 360/num_slices
      x1 = center[0] + l * np.sin(np.radians(startAngle))
      y1 = center[1] - l * np.cos(np.radians(startAngle))
      x2 = center[0] + l * np.sin(np.radians(endAngle))
      y2 = center[1] - l * np.cos(np.radians(endAngle))
      vertices = [(center[0], center[1]), (y1, x1), (y2, x2)]
      # Make empt black canvas
      # Draw this pie slice in white
      cv2.fillPoly(im, np.array([vertices],'int32'), 255)
  return cv2.GaussianBlur(im, (5,5), cv2.COLOR_RGB2GRAY )
def apply_mask(img, center, radius):
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    cv2.GaussianBlur(mask, (0, 0), radius, mask)

    mask = mask.astype(np.float32) / 255.0
    #get brightest pixel in mask
    max = np.amax(mask)
    mask *= 1 / max
    result = (img/255.)*(mask/255)*255

    return result
def get_filters(shape, center):
    filters=[
        generate_sinusoidal_gradient(shape, center, 0.1, 1),
    ]
def find_symmetries(img):
  
    #Resize image to 128x128
    original_size = (img.shape[1], img.shape[0])
    working_width = 512
    working_height = int(img.shape[0] * working_width / img.shape[1])
    img = cv2.resize(img, (working_width, working_height))
    center_offset = (0, 0)
    center_offset = (center_offset[0] * working_width / original_size[0], center_offset[1] * 128 / working_height)

    # Convert input image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply a Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1)
    #get width and height of prev_img_cv2
    width = img.shape[1]
    height = img.shape[0]
    offset_from_center = (int(width//2-center_offset[0]),
                           int(height//2-center_offset[1]))
    #apply mask to prev_img_cv2
    # Compute the negative Laplacian of the image using a LoG filter
    log_filter = cv2.Laplacian(img_blur, cv2.CV_64F)
    log_filter = -log_filter



    masked = apply_mask(log_filter,offset_from_center , 16)
    return log_filter

def sum_along_axis(img, sum_area_width, axis):
    full_axis = img.shape[axis]
    cropped_axis=img.shape[1-axis]

    crop = img



def prep_img(img, expected_vanishing_point_center_offset, crop_size):
    original_size = (img.shape[0], img.shape[1])
    working_width = 256
    working_height = int(img.shape[0] * working_width / img.shape[1])
    img = cv2.resize(img, (working_width, working_height))
    center_offset = expected_vanishing_point_center_offset
    width = img.shape[0]
    height = img.shape[1]
    center_offset = (center_offset[0] * working_width / original_size[0], center_offset[1] * working_height / original_size[1])
    vanishing_point = (int(width//2-center_offset[0]),
                             int(height//2-center_offset[1]))
    
    # Convert input image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 1)
    
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #get width and height of prev_img_cv2
    width = img.shape[0]
    height = img.shape[1]

    #crop out crop_size around vanishing point
    crop = img[vanishing_point[1]-crop_size//2:vanishing_point[1]+crop_size//2, vanishing_point[0]-crop_size//2:vanishing_point[0]+crop_size//2]

    return crop

def sum_pixels_in_box(image, axis, offset):
    # Determine the center of the image along the specified axis
    offaxis=image.shape[1-axis]
    offaxisCenter = offaxis // 2
    
    # Determine the range of indices to sum over
    startOffaxis = offaxisCenter-offset
    endOffaxis = offaxisCenter+offset
    startAxis=0
    endAxis=image.shape[axis]

    if(axis==0):
        area=image[startAxis:endAxis, startOffaxis:endOffaxis]
    else:
        area=image[startOffaxis:endOffaxis, startAxis:endAxis]

    # Sum the pixels in the specified box
    return np.sum(area)

def get_symmetry_realignment(img, expected_vanishing_point_center_offset, axis):
    roam_range=16
    roam_step=1
    crop_size=64
    img = prep_img(img, expected_vanishing_point_center_offset, crop_size+2*roam_range)
    
    minSum = 100000000000
    bestShift = 0
    bestImg=None
    display=[]

    for x in range(-roam_range, roam_range, roam_step):
      shiftImg = np.roll(img, x, axis=1-axis)
      blurKernel=(5,5)
      img=cv2.GaussianBlur(img, blurKernel, sigmaX=1)
      shiftImg=cv2.GaussianBlur(shiftImg, blurKernel, sigmaX=1)

      diff = cv2.absdiff(img,np.flip(shiftImg, axis=1-axis))
      
      #Sobel
      #diff = cv2.Sobel(diff,cv2.CV_64F,1,0,ksize=3)
      #diff=cv2.Laplacian(diff, cv2.CV_64F)
      #diff = np.absolute(diff)
      #diff = np.uint8(diff)

      centerX=diff.shape[0]//2
      centerY= diff.shape[1]//2
      if axis == 0: 
        cropped = diff[ centerY-crop_size//2:centerY+crop_size//2,
                       centerX-crop_size//2-x:centerX+crop_size//2-x]
      else:
        cropped = diff[ centerX-crop_size//2-x:centerX+crop_size//2-x,
                       centerY-crop_size//2:centerY+crop_size//2]
        
      cv2.normalize(cropped, cropped, 0, 255, cv2.NORM_MINMAX)

      sum = sum_pixels_in_box(cropped, axis, 2)
      display.append((cropped, sum))
      if sum < minSum:
          minSum = sum
          bestShift = x
          bestImg = cropped
    return bestShift
    

def find_line_of_symmetry(img, expected_vanishing_point_center_offset, axis):
    img = prep_img(img, expected_vanishing_point_center_offset, 64)
    # get the sum of one row of pixels
    sums = np.sum(img, axis=axis)
    #split sums into two halve
    halves=[
            (sums[:len(sums)//2], np.sum(left)),
            (sums[len(sums)//2:], np.sum(right))
            ]
    halves.sort(key=lambda x: x[1])
    while(halves[0] < halves[1]):
        #move one item from halves[0][0] to halves[1][0]
        move = halves[1][0].pop(0)
        halves[1][1] -= move
        halves[0][0].append(move)
        halves[0][1] += move
    mostSymmetricAt = len(halves[0][0])

def easeInOutQuad(x):
    return 2 * x ** 2 if x < 0.5 else 1 - ((-2 * x + 2) ** 2) / 2


    
            
def find_best_alignment(img, prev_img, expected_vanishing_point_center_offset):
  crop_size = 64
  test_line_of_symmetry(img, expected_vanishing_point_center_offset, 1)
  test_line_of_symmetry(img, expected_vanishing_point_center_offset, 0)
  c1 = prep_img(img, expected_vanishing_point_center_offset, crop_size)
  c2 = prep_img(prev_img, expected_vanishing_point_center_offset, crop_size)
  epch =  int(time.time())

  roam_range=4
#create new image of size crop_zize*2+roam_range, to hold all possible alignments
    #loop through all possible alignments
  minDifference = 10000000000
  bestAlignment = (0, 0)
  bestImg = np.copy(c2)
  for x in range(-roam_range, roam_range):
    for y in range(-roam_range, roam_range):
        img = np.copy(c2)
        #shift prev_img by i,j
        shifted = np.roll(c1, x, axis=0)
        shifted = np.roll(shifted, y, axis=1)
        #add shifted prev_img to new image
        img = img - shifted
        #crop out center crop_size
        #find best alignment
           
        img = img[crop_size-roam_range:crop_size+roam_range, crop_size-roam_range:crop_size+roam_range]
        difference = np.sum(np.abs(img))
        

        if difference < minDifference:
            minDifference = difference
            bestAlignment = (x, y)
            print("New best alignment!")
            bestImg = img
   #save c1 & c2 to C:\Code\stable-diffusion\outputs\dbg

  print("Best alignment: ", bestAlignment)
  return bestAlignment

    
    
lastN = 13
symmetryTargetOffsets=[]
def align_vanishing_point(img, prev_img, expected_vanishing_point_center_offset,frame_idx):
    global symmetryTargetOffsets
    global lastN
    #find current offset to target by pixels
    if len(symmetryTargetOffsets) == 0:
        symmetryTargetOffsets.append(expected_vanishing_point_center_offset)
        symmetryTargetOffsets *= lastN-1

    circles = find_circles(img)
    vanishing_point = (img.shape[0]//2-expected_vanishing_point_center_offset[0], img.shape[1]//2-expected_vanishing_point_center_offset[1])
    bestAlignment = get_hough_target_point(img, circles, vanishing_point,frame_idx)
    if bestAlignment is not None:
        bestAlignment = (bestAlignment[0]-img.shape[0]//2,img.shape[1]//2-bestAlignment[1])
        symmetryTargetOffsets.append(bestAlignment)
    else:
        symmetryTargetOffsets.append((0,0))
    
    #calculate average offset
    return get_translation_speed()
    #return (0,0)

def get_translation_speed():
    global symmetryTargetOffsets


    avgOffset = np.mean(np.array(symmetryTargetOffsets[-lastN:]), axis=0)
    speedFactor = 0.08
    return [ speedFactor/lastN * x for x in avgOffset]

def get_hough_target_point(img,circles, vanishingPoint, frame_idx):
    global symmetryTargetOffsets


    dbgImage=np.copy(img)
    maxDistance=min(1.1**(-frame_idx+100)+100, 300)
    nearestCircle = None
    nearestRadius=None
    nearestIsChain = False
    nearestCircleDistance = 100000
    if(circles is None):
        return nearestCircle
    for i in circles[0, :]:
        circleCenter = (i[0] , i[1])
        radius = i[2]
        #calculate distance between imageCenter and circle center
        distanceToVanishingPoint = np.linalg.norm(np.array(circleCenter) - np.array(vanishingPoint))
        prevCircle = next((x for x in symmetryTargetOffsets[::-1] if x is not (0,0) and x is not None), vanishingPoint)
        distanceToPreviousCircle = np.linalg.norm(np.array(circleCenter) - np.array(prevCircle))

        #Discard all circles that jumpt to quickly
        maxMovementPerFrame = 20
        numberOfFramesSincePreviousCircle = len(symmetryTargetOffsets) - symmetryTargetOffsets.index(prevCircle) + 1

        cv2.circle(dbgImage, vanishingPoint, maxMovementPerFrame * numberOfFramesSincePreviousCircle, (128, 128, 128), 1)
        if (distanceToVanishingPoint > maxDistance and distanceToPreviousCircle > maxDistance ):
            cv2.circle(dbgImage, circleCenter, 1, (255, 255, 255), 3)
            cv2.circle(dbgImage, circleCenter, radius, (128, 128, 128), 1)
            continue
        #The nerest circle is the one closest to the vanishing point
        if  distanceToVanishingPoint < nearestCircleDistance:
            nearestCircle = circleCenter
            nearestCircleDistance = distanceToVanishingPoint
            nearestIsChain=False
            nearestRadius = radius
        #The nearest circle is the one closest to the previous circle
        if  distanceToPreviousCircle < nearestCircleDistance and distanceToPreviousCircle < maxMovementPerFrame * numberOfFramesSincePreviousCircle:
            nearestCircle = circleCenter
            nearestCircleDistance = distanceToPreviousCircle
            nearestRadius = radius
            nearestIsChain = True
    epch= int(time.time())

    if(nearestCircle is None):
        cv2.imwrite("C:\\Code\\stable-diffusion\\outputs\\dbg\\circles"+str(epch)+".png", dbgImage)
        return nearestCircle
    
    if(nearestIsChain):
        cv2.circle(dbgImage, nearestCircle, 1, (0, 0, 0), 3)
    else:
        cv2.circle(dbgImage, nearestCircle, 1, (255, 255, 255), 3)

    cv2.circle(dbgImage, nearestCircle, nearestRadius, (0, 255, 0), 3)
    print("Nearest circle: ", nearestCircle)
    epch= int(time.time())
    cv2.imwrite("C:\\Code\\stable-diffusion\\outputs\\dbg\\circles"+str(epch)+".png", dbgImage)

    return nearestCircle

def find_circles(img):
    working_width = 512
    working_height = int(img.shape[0] * working_width / img.shape[1])
    prepped = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    prepped = cv2.resize(prepped, (working_width, working_height))
    prepped = cv2.GaussianBlur(prepped, (5,5), sigmaX=1)
    rows = img.shape[0]
    circles = cv2.HoughCircles(prepped, cv2.HOUGH_GRADIENT, 1, rows / 4,
                               param1=100, param2=45,
                               minRadius=1, maxRadius=64)
    if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        # circle outline
        radius = i[2]
        cv2.circle(prepped, center, 1, (0, 100, 100), 3)
        cv2.circle(prepped, center, radius, (255, 0, 255), 3)
        i[0] = int(i[0] * (img.shape[1] / working_width))
        i[1] = int(i[1]*(img.shape[0] / working_height))
        i[2] = i[2]*(img.shape[1] / working_width)
    return circles

def mark_point(image, point, color):
    x, y = point
    b, g, r = color
    #marked_image = cv2.circle(image, (x, y), 5, (int(b), int(g), int(r)), -1)
    return image

