
import torch
import numpy as np
import os

from os.path import join, isdir, isfile, expanduser
from PIL import Image

from torchvision import transforms
from torchvision.transforms.transforms import Resize

from torch.nn import functional as nnf
from general_utils import get_from_repository

from skimage.draw import polygon2mask



def random_crop_slices(origin_size, target_size):
    """Gets slices of a random crop. """
    assert origin_size[0] >= target_size[0] and origin_size[1] >= target_size[1], f'actual size: {origin_size}, target size: {target_size}'

    offset_y = torch.randint(0, origin_size[0] - target_size[0] + 1, (1,)).item()  # range: 0 <= value < high
    offset_x = torch.randint(0, origin_size[1] - target_size[1] + 1, (1,)).item()

    return slice(offset_y, offset_y + target_size[0]), slice(offset_x, offset_x + target_size[1])


def find_crop(seg, image_size, iterations=1000, min_frac=None, best_of=None):


    best_crops = []
    best_crop_not_ok = float('-inf'), None, None
    min_sum = 0

    seg = seg.astype('bool')
    
    if min_frac is not None:
        #min_sum = seg.sum() * min_frac
        min_sum = seg.shape[0] * seg.shape[1] * min_frac
    
    for iteration in range(iterations):
        sl_y, sl_x = random_crop_slices(seg.shape, image_size)
        seg_ = seg[sl_y, sl_x]
        sum_seg_ = seg_.sum()

        if sum_seg_ > min_sum:

            if best_of is None:
                return sl_y, sl_x, False
            else:
                best_crops += [(sum_seg_, sl_y, sl_x)]
                if len(best_crops) >= best_of:
                    best_crops.sort(key=lambda x:x[0], reverse=True)
                    sl_y, sl_x = best_crops[0][1:]
                    
                    return sl_y, sl_x, False

        else:
            if sum_seg_ > best_crop_not_ok[0]:
                best_crop_not_ok = sum_seg_, sl_y, sl_x
        
    else:
        # return best segmentation found
        return best_crop_not_ok[1:] + (best_crop_not_ok[0] <= min_sum,) 


class PhraseCut(object):

    def __init__(self, split, image_size=400, negative_prob=0, aug=None, aug_color=False, aug_crop=True,
                 min_size=0, remove_classes=None, with_visual=False, only_visual=False, mask=None):
        super().__init__()

        self.negative_prob = negative_prob
        self.image_size = image_size
        self.with_visual = with_visual
        self.only_visual = only_visual
        self.phrase_form = '{}'
        self.mask = mask
        self.aug_crop = aug_crop
        
        if aug_color:
            self.aug_color = transforms.Compose([
                transforms.ColorJitter(0.5, 0.5, 0.2, 0.05),
            ])
        else:
            self.aug_color = None

        get_from_repository('PhraseCut', ['PhraseCut.tar'], integrity_check=lambda local_dir: all([
            isdir(join(local_dir, 'VGPhraseCut_v0')),
            isdir(join(local_dir, 'VGPhraseCut_v0', 'images')),
            isfile(join(local_dir, 'VGPhraseCut_v0', 'refer_train.json')),
            len(os.listdir(join(local_dir, 'VGPhraseCut_v0', 'images'))) in {108250, 108249}
        ]))

        from third_party.PhraseCutDataset.utils.refvg_loader import RefVGLoader
        self.refvg_loader = RefVGLoader(split=split)

        # img_ids where the size in the annotations does not match actual size
        invalid_img_ids = set([150417, 285665, 498246, 61564, 285743, 498269, 498010, 150516, 150344, 286093, 61530, 
                               150333, 286065, 285814, 498187, 285761, 498042])
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std)

        self.sample_ids = [(i, j) 
                           for i in self.refvg_loader.img_ids 
                           for j in range(len(self.refvg_loader.get_img_ref_data(i)['phrases']))
                           if i not in invalid_img_ids]
        

        # self.all_phrases = list(set([p for i in self.refvg_loader.img_ids for p in self.refvg_loader.get_img_ref_data(i)['phrases']]))

        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()        

        # Filter by class (if remove_classes is set)
        if remove_classes is None:
            pass
        else:
            from datasets.generate_lvis_oneshot import PASCAL_SYNSETS, traverse_lemmas, traverse_lemmas_hypo
            from nltk.corpus import wordnet

            print('remove pascal classes...')

            get_data = self.refvg_loader.get_img_ref_data  # shortcut
            keep_sids = None

            if remove_classes[0] == 'pas5i':
                subset_id = remove_classes[1]
                from datasets.generate_lvis_oneshot import PASCAL_5I_SYNSETS_ORDERED, PASCAL_5I_CLASS_IDS
                avoid = [PASCAL_5I_SYNSETS_ORDERED[i] for i in range(20) if i+1 not in PASCAL_5I_CLASS_IDS[subset_id]]
      

            elif remove_classes[0] == 'zs':
                stop = remove_classes[1]
                
                from datasets.pascal_zeroshot import PASCAL_VOC_CLASSES_ZS

                avoid = [c for class_set in PASCAL_VOC_CLASSES_ZS[:stop] for c in class_set]
                print(avoid)

            elif remove_classes[0] == 'aff':
                # avoid = ['drink.v.01', 'sit.v.01', 'ride.v.02']
                # all_lemmas = set(['drink', 'sit', 'ride'])
                avoid = ['drink', 'drinks', 'drinking', 'sit', 'sits', 'sitting', 
                         'ride', 'rides', 'riding',
                         'fly', 'flies', 'flying', 'drive', 'drives', 'driving', 'driven', 
                         'swim', 'swims', 'swimming',
                         'wheels', 'wheel', 'legs', 'leg', 'ear', 'ears']
                keep_sids = [(i, j) for i, j in self.sample_ids if 
                             all(x not in avoid for x in get_data(i)['phrases'][j].split(' '))]

            print('avoid classes:', avoid)


            if keep_sids is None:
                all_lemmas = [s for ps in avoid for s in traverse_lemmas_hypo(wordnet.synset(ps), max_depth=None)]
                all_lemmas = list(set(all_lemmas))
                all_lemmas = [h.replace('_', ' ').lower() for h in all_lemmas]
                all_lemmas = set(all_lemmas)

                # divide into multi word and single word
                all_lemmas_s = set(l for l in all_lemmas if ' ' not in l)
                all_lemmas_m = set(l for l in all_lemmas if l not in all_lemmas_s)

                # new3
                phrases = [get_data(i)['phrases'][j] for i, j in self.sample_ids]
                remove_sids = set((i,j) for (i,j), phrase in zip(self.sample_ids, phrases)
                                  if any(l in phrase for l in all_lemmas_m) or 
                                  len(set(wnl.lemmatize(w) for w in phrase.split(' ')).intersection(all_lemmas_s)) > 0
                )
                keep_sids = [(i, j) for i, j in self.sample_ids if (i,j) not in remove_sids]

            print(f'Reduced to {len(keep_sids) / len(self.sample_ids):.3f}')
            removed_ids = set(self.sample_ids) - set(keep_sids)

            print('Examples of removed', len(removed_ids))
            for i, j in list(removed_ids)[:20]:
                print(i, get_data(i)['phrases'][j])

            self.sample_ids = keep_sids

        from itertools import groupby
        samples_by_phrase = [(self.refvg_loader.get_img_ref_data(i)['phrases'][j], (i, j)) 
                             for i, j in self.sample_ids]
        samples_by_phrase = sorted(samples_by_phrase)
        samples_by_phrase = groupby(samples_by_phrase, key=lambda x: x[0])
        
        self.samples_by_phrase = {prompt: [s[1] for s in prompt_sample_ids] for prompt, prompt_sample_ids in samples_by_phrase}

        self.all_phrases = list(set(self.samples_by_phrase.keys()))


        if self.only_visual:
            assert self.with_visual
            self.sample_ids = [(i, j) for i, j in self.sample_ids
                               if len(self.samples_by_phrase[self.refvg_loader.get_img_ref_data(i)['phrases'][j]]) > 1]

        # Filter by size (if min_size is set)
        sizes = [self.refvg_loader.get_img_ref_data(i)['gt_boxes'][j] for i, j in self.sample_ids]
        image_sizes = [self.refvg_loader.get_img_ref_data(i)['width'] * self.refvg_loader.get_img_ref_data(i)['height'] for i, j in self.sample_ids]
        #self.sizes = [sum([(s[2] - s[0]) * (s[3] - s[1]) for s in size]) for size in sizes]
        self.sizes = [sum([s[2] * s[3] for s in size]) / img_size for size, img_size in zip(sizes, image_sizes)]

        if min_size:
            print('filter by size')

        self.sample_ids = [self.sample_ids[i] for i in range(len(self.sample_ids)) if self.sizes[i] > min_size]

        self.base_path = join(expanduser('~/datasets/PhraseCut/VGPhraseCut_v0/images/'))

    def __len__(self):
        return len(self.sample_ids)


    def load_sample(self, sample_i, j):

        img_ref_data = self.refvg_loader.get_img_ref_data(sample_i)

        polys_phrase0 = img_ref_data['gt_Polygons'][j]
        phrase = img_ref_data['phrases'][j]
        phrase = self.phrase_form.format(phrase)

        masks = []
        for polys in polys_phrase0:
            for poly in polys:
                poly = [p[::-1] for p in poly]  # swap x,y
                masks += [polygon2mask((img_ref_data['height'], img_ref_data['width']), poly)]

        seg = np.stack(masks).max(0)
        img = np.array(Image.open(join(self.base_path, str(img_ref_data['image_id']) + '.jpg')))

        min_shape = min(img.shape[:2])

        if self.aug_crop:
            sly, slx, exceed = find_crop(seg, (min_shape, min_shape), iterations=50, min_frac=0.05)
        else:
            sly, slx = slice(0, None), slice(0, None)
    
        seg = seg[sly, slx]
        img = img[sly, slx]

        seg = seg.astype('uint8')
        seg = torch.from_numpy(seg).view(1, 1, *seg.shape)

        if img.ndim == 2:
            img = np.dstack([img] * 3)

        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()

        seg = nnf.interpolate(seg, (self.image_size, self.image_size), mode='nearest')[0,0]
        img = nnf.interpolate(img, (self.image_size, self.image_size), mode='bilinear', align_corners=True)[0]

        # img = img.permute([2,0, 1])
        img = img / 255.0

        if self.aug_color is not None:
            img = self.aug_color(img)

        img = self.normalize(img)



        return img, seg, phrase

    def __getitem__(self, i):
 
        sample_i, j = self.sample_ids[i]

        img, seg, phrase = self.load_sample(sample_i, j)

        if self.negative_prob > 0:
            if torch.rand((1,)).item() < self.negative_prob:

                new_phrase = None
                while new_phrase is None or new_phrase == phrase:
                    idx = torch.randint(0, len(self.all_phrases), (1,)).item()
                    new_phrase = self.all_phrases[idx]
                phrase = new_phrase
                seg = torch.zeros_like(seg)

        if self.with_visual:
            # find a corresponding visual image
            if phrase in self.samples_by_phrase and len(self.samples_by_phrase[phrase]) > 1:
                idx = torch.randint(0, len(self.samples_by_phrase[phrase]), (1,)).item()
                other_sample = self.samples_by_phrase[phrase][idx]
                #print(other_sample)
                img_s, seg_s, _ = self.load_sample(*other_sample)

                from datasets.utils import blend_image_segmentation

                if self.mask in {'separate', 'text_and_separate'}:
                    # assert img.shape[1:] == img_s.shape[1:] == seg_s.shape == seg.shape[1:]
                    add_phrase = [phrase] if self.mask == 'text_and_separate' else []
                    vis_s = add_phrase + [img_s, seg_s, True]
                else:
                    if self.mask.startswith('text_and_'):
                        mask_mode = self.mask[9:]
                        label_add = [phrase]
                    else:
                        mask_mode = self.mask
                        label_add = []

                    masked_img_s = torch.from_numpy(blend_image_segmentation(img_s, seg_s, mode=mask_mode, image_size=self.image_size)[0])
                    vis_s = label_add + [masked_img_s, True]
                
            else:
                # phrase is unique
                vis_s = torch.zeros_like(img)

                if self.mask in {'separate', 'text_and_separate'}:
                    add_phrase = [phrase] if self.mask == 'text_and_separate' else []
                    vis_s = add_phrase + [vis_s, torch.zeros(*vis_s.shape[1:], dtype=torch.uint8), False]
                elif self.mask.startswith('text_and_'):
                    vis_s = [phrase, vis_s, False]
                else:
                    vis_s = [vis_s, False]
        else:
            assert self.mask == 'text'
            vis_s = [phrase]
        
        seg = seg.unsqueeze(0).float()

        data_x = (img,) + tuple(vis_s)

        return data_x, (seg, torch.zeros(0), i)


class PhraseCutPlus(PhraseCut):

    def __init__(self, split, image_size=400, aug=None, aug_color=False, aug_crop=True, min_size=0, remove_classes=None, only_visual=False, mask=None):
        super().__init__(split, image_size=image_size, negative_prob=0.2, aug=aug, aug_color=aug_color, aug_crop=aug_crop, min_size=min_size, 
                         remove_classes=remove_classes, with_visual=True, only_visual=only_visual, mask=mask)