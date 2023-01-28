from torch.functional import Tensor
from general_utils import load_model
from torch.utils.data import DataLoader
import torch
import numpy as np

def denorm(img):

    np_input = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        np_input = True

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    img_denorm = (img*std[:,None,None]) + mean[:,None,None]

    if np_input:
        img_denorm = np.clip(img_denorm.numpy(), 0, 1)
    else:
        img_denorm = torch.clamp(img_denorm, 0, 1)

    return img_denorm


def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (img - mean[:,None,None]) / std[:,None,None]


def fast_iou_curve(p, g):
    
    g = g[p.sort().indices]
    p = torch.sigmoid(p.sort().values)
    
    scores = []
    vals = np.linspace(0, 1, 50)

    for q in vals:

        n = int(len(g) * q)

        valid = torch.where(p > q)[0]
        if len(valid) > 0:
            n = int(valid[0])
        else:
            n = len(g)

        fn = g[:n].sum()
        tn = n - fn
        tp = g[n:].sum()
        fp = len(g) - n - tp

        iou = tp / (tp + fn + fp)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        scores += [iou]
        
    return vals, scores


def fast_rp_curve(p, g):
    
    g = g[p.sort().indices]
    p = torch.sigmoid(p.sort().values)
    
    precisions, recalls = [], []
    vals = np.linspace(p.min(), p.max(), 250)

    for q in p[::100000]:

        n = int(len(g) * q)

        valid = torch.where(p > q)[0]
        if len(valid) > 0:
            n = int(valid[0])
        else:
            n = len(g)

        fn = g[:n].sum()
        tn = n - fn
        tp = g[n:].sum()
        fp = len(g) - n - tp

        iou = tp / (tp + fn + fp)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        precisions += [precision]
        recalls += [recall]
        
    return recalls, precisions


# Image processing

def img_preprocess(batch, blur=0, grayscale=False, center_context=None, rect=False, rect_color=(255,0,0), rect_width=2, 
                   brightness=1.0, bg_fac=1, colorize=False, outline=False, image_size=224):
    import cv2

    rw = rect_width

    out = []
    for img, mask in zip(batch[1], batch[2]):

        img = img.cpu() if isinstance(img, torch.Tensor) else torch.from_numpy(img)
        mask = mask.cpu() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)
        
        img *= brightness
        img_bl = img
        if blur > 0: # best 5
            img_bl = torch.from_numpy(cv2.GaussianBlur(img.permute(1,2,0).numpy(), (15, 15), blur)).permute(2,0,1)
        
        if grayscale:
            img_bl = img_bl[1][None]
        
        #img_inp = img_ratio*img*mask + (1-img_ratio)*img_bl
        # img_inp = img_ratio*img*mask + (1-img_ratio)*img_bl * (1-mask)
        img_inp = img*mask + (bg_fac) * img_bl * (1-mask)

        if rect:
            _, bbox = crop_mask(img, mask, context=0.1)
            img_inp[:, bbox[2]: bbox[3], max(0, bbox[0]-rw):bbox[0]+rw] = torch.tensor(rect_color)[:,None,None]
            img_inp[:, bbox[2]: bbox[3], max(0, bbox[1]-rw):bbox[1]+rw] = torch.tensor(rect_color)[:,None,None]
            img_inp[:, max(0, bbox[2]-1): bbox[2]+rw, bbox[0]:bbox[1]] = torch.tensor(rect_color)[:,None,None]
            img_inp[:, max(0, bbox[3]-1): bbox[3]+rw, bbox[0]:bbox[1]] = torch.tensor(rect_color)[:,None,None]


        if center_context is not None:
            img_inp = object_crop(img_inp, mask, context=center_context, image_size=image_size)

        if colorize:
            img_gray = denorm(img)
            img_gray = cv2.cvtColor(img_gray.permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY)
            img_gray = torch.stack([torch.from_numpy(img_gray)]*3)
            img_inp = torch.tensor([1,0.2,0.2])[:,None,None] * img_gray * mask + bg_fac * img_gray * (1-mask)
            img_inp = norm(img_inp)

        if outline:
            cont = cv2.findContours(mask.byte().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline_img = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(outline_img, cont[0], -1, thickness=5, color=(255, 255, 255))
            outline_img = torch.stack([torch.from_numpy(outline_img)]*3).float() / 255.
            img_inp = torch.tensor([1,0,0])[:,None,None] *  outline_img + denorm(img_inp) * (1- outline_img)
            img_inp = norm(img_inp)

        out += [img_inp]

    return torch.stack(out)


def object_crop(img, mask, context=0.0, square=False, image_size=224):
    img_crop, bbox = crop_mask(img, mask, context=context, square=square)
    img_crop = pad_to_square(img_crop, channel_dim=0)
    img_crop = torch.nn.functional.interpolate(img_crop.unsqueeze(0), (image_size, image_size)).squeeze(0)
    return img_crop
    

def crop_mask(img, mask, context=0.0, square=False):
    
    assert img.shape[1:] == mask.shape
    
    bbox = [mask.max(0).values.argmax(), mask.size(0) - mask.max(0).values.flip(0).argmax()]
    bbox += [mask.max(1).values.argmax(), mask.size(1) - mask.max(1).values.flip(0).argmax()]
    bbox = [int(x) for x in bbox]
    
    width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])

    # square mask
    if square:
        bbox[0] = int(max(0, bbox[0] - context * height))
        bbox[1] = int(min(mask.size(0), bbox[1] + context * height))
        bbox[2] = int(max(0, bbox[2] - context * width))
        bbox[3] = int(min(mask.size(1), bbox[3] + context * width))

        width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])
        if height > width:
            bbox[2] = int(max(0, (bbox[2] - 0.5*height)))
            bbox[3] = bbox[2] + height
        else:
            bbox[0] = int(max(0, (bbox[0] - 0.5*width)))
            bbox[1] = bbox[0] + width
    else:
        bbox[0] = int(max(0, bbox[0] - context * height))
        bbox[1] = int(min(mask.size(0), bbox[1] + context * height))
        bbox[2] = int(max(0, bbox[2] - context * width))
        bbox[3] = int(min(mask.size(1), bbox[3] + context * width))

    width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])
    img_crop = img[:, bbox[2]: bbox[3], bbox[0]: bbox[1]]
    return img_crop, bbox


def pad_to_square(img, channel_dim=2, fill=0):
    """


    add padding such that a squared image is returned """
    
    from torchvision.transforms.functional import pad

    if channel_dim == 2:
        img = img.permute(2, 0, 1)
    elif channel_dim == 0:
        pass
    else:
        raise ValueError('invalid channel_dim')

    h, w = img.shape[1:]
    pady1 = pady2 = padx1 = padx2 = 0

    if h > w:
        padx1 = (h - w) // 2
        padx2 = h - w - padx1
    elif w > h:
        pady1 = (w - h) // 2
        pady2 = w - h - pady1

    img_padded = pad(img, padding=(padx1, pady1, padx2, pady2), padding_mode='constant')

    if channel_dim == 2:
        img_padded = img_padded.permute(1, 2, 0)

    return img_padded


# qualitative

def split_sentence(inp, limit=9):
    t_new, current_len = [], 0
    for k, t in enumerate(inp.split(' ')):
        current_len += len(t) + 1
        t_new += [t+' ']
        # not last
        if current_len > limit and k != len(inp.split(' ')) - 1:
            current_len = 0
            t_new += ['\n']

    t_new = ''.join(t_new)
    return t_new
    

from matplotlib import pyplot as plt


def plot(imgs, *preds, labels=None, scale=1, cmap=plt.cm.magma, aps=None, gt_labels=None, vmax=None):
    
    row_off = 0 if labels is None else 1
    _, ax = plt.subplots(len(imgs) + row_off, 1 + len(preds), figsize=(scale * float(1 + 2*len(preds)), scale * float(len(imgs)*2)))
    [a.axis('off') for a in ax.flatten()]
    
    if labels is not None:
        for j in range(len(labels)):
            t_new = split_sentence(labels[j], limit=6)
            ax[0, 1+ j].text(0.5, 0.1, t_new, ha='center', fontsize=3+ 10*scale)


    for i in range(len(imgs)):
        ax[i + row_off,0].imshow(imgs[i])
        for j in range(len(preds)):
            img = preds[j][i][0].detach().cpu().numpy()

            if gt_labels is not None and labels[j] == gt_labels[i]:
                print(j, labels[j], gt_labels[i])
                edgecolor = 'red'
                if aps is not None:
                    ax[i + row_off, 1 + j].text(30, 70, f'AP: {aps[i]:.3f}', color='red', fontsize=8)
            else:
                edgecolor = 'k'

            rect = plt.Rectangle([0,0], img.shape[0], img.shape[1], facecolor="none", 
                                 edgecolor=edgecolor, linewidth=3)
            ax[i + row_off,1 + j].add_patch(rect)

            if vmax is None:
                this_vmax = 1 
            elif vmax == 'per_prompt':
                this_vmax = max([preds[j][_i][0].max() for _i in range(len(imgs))])
            elif vmax == 'per_image':
                this_vmax = max([preds[_j][i][0].max() for _j in range(len(preds))])

            ax[i + row_off,1 + j].imshow(img, vmin=0, vmax=this_vmax, cmap=cmap)

    
            # ax[i,1 + j].imshow(preds[j][i][0].detach().cpu().numpy(), vmin=preds[j].min(), vmax=preds[j].max())
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)    