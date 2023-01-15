
import numpy as np
import torch


def blend_image_segmentation(img, seg, mode, image_size=224):


    if mode in {'blur_highlight', 'blur3_highlight', 'blur3_highlight01', 'blur_highlight_random', 'crop'}:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        if isinstance(seg, np.ndarray):
            seg = torch.from_numpy(seg)            

    if mode == 'overlay':
        out = img * seg
        out = [out.astype('float32')]
    elif mode == 'highlight':
        out = img * seg[None, :, :] * 0.85 + 0.15 * img
        out = [out.astype('float32')]
    elif mode == 'highlight2':
        img = img / 2
        out = (img+0.1) * seg[None, :, :] + 0.3 * img
        out = [out.astype('float32')]
    elif mode == 'blur_highlight':
        from evaluation_utils import img_preprocess
        out  = [img_preprocess((None, [img], [seg]), blur=1, bg_fac=0.5).numpy()[0] - 0.01]
    elif mode == 'blur3_highlight':
        from evaluation_utils import img_preprocess
        out  = [img_preprocess((None, [img], [seg]), blur=3, bg_fac=0.5).numpy()[0] - 0.01]
    elif mode == 'blur3_highlight01':
        from evaluation_utils import img_preprocess
        out  = [img_preprocess((None, [img], [seg]), blur=3, bg_fac=0.1).numpy()[0] - 0.01]                
    elif mode == 'blur_highlight_random':
        from evaluation_utils import img_preprocess
        out  = [img_preprocess((None, [img], [seg]), blur=0 + torch.randint(0, 3, (1,)).item(), bg_fac=0.1 + 0.8*torch.rand(1).item()).numpy()[0] - 0.01]               
    elif mode == 'crop':
        from evaluation_utils import img_preprocess
        out  = [img_preprocess((None, [img], [seg]), blur=1, center_context=0.1, image_size=image_size)[0].numpy()]  
    elif mode == 'crop_blur_highlight':
        from evaluation_utils import img_preprocess
        out  = [img_preprocess((None, [img], [seg]), blur=3, center_context=0.1, bg_fac=0.1, image_size=image_size)[0].numpy()]  
    elif mode == 'crop_blur_highlight352':
        from evaluation_utils import img_preprocess
        out  = [img_preprocess((None, [img], [seg]), blur=3, center_context=0.1, bg_fac=0.1, image_size=352)[0].numpy()]          
    elif mode == 'shape':
        out = [np.stack([seg[:, :]]*3).astype('float32')]
    elif mode == 'concat':
        out = [np.concatenate([img, seg[None, :, :]]).astype('float32')]
    elif mode == 'image_only':
        out = [img.astype('float32')]
    elif mode == 'image_black':
        out = [img.astype('float32')*0]        
    elif mode is None:
        out = [img.astype('float32')]
    elif mode == 'separate':
        out = [img.astype('float32'), seg.astype('int64')]
    elif mode == 'separate_img_black':
        out = [img.astype('float32')*0, seg.astype('int64')]        
    elif mode == 'separate_seg_ones':
        out = [img.astype('float32'), np.ones_like(seg).astype('int64')]                
    elif mode == 'separate_both_black':
        out = [img.astype('float32')*0, seg.astype('int64')*0]        
    else:
        raise ValueError(f'invalid mode: {mode}')

    return out