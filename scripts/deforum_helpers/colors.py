import cv2
import pkg_resources
from skimage.exposure import match_histograms

def maintain_colors(prev_img, color_match_sample, mode):
    skimage_version = pkg_resources.get_distribution('scikit-image').version
    is_skimage_v20_or_higher = pkg_resources.parse_version(skimage_version) >= pkg_resources.parse_version('0.20.0')
    
    match_histograms_kwargs = {'channel_axis': -1} if is_skimage_v20_or_higher else {'multichannel': True}
    
    if mode == 'RGB':
        return match_histograms(prev_img, color_match_sample, **match_histograms_kwargs)
    elif mode == 'HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, **match_histograms_kwargs)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, **match_histograms_kwargs)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
