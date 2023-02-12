from os.path import expanduser
import torch
import json
from general_utils import get_from_repository
from datasets.lvis_oneshot3 import blend_image_segmentation
from general_utils import log

PASCAL_CLASSES = {a['id']: a['synonyms'] for a in json.load(open('datasets/pascal_classes.json'))}


class PFEPascalWrapper(object):

    def __init__(self, mode, split, mask='separate', image_size=473, label_support=None, size=None, p_negative=0, aug=None):
        import sys
        # sys.path.append(expanduser('~/projects/new_one_shot'))
        from third_party.PFENet.util.dataset import SemData

        get_from_repository('PascalVOC2012', ['Pascal5i.tar'])

        self.p_negative = p_negative
        self.size = size
        self.mode = mode
        self.image_size = image_size
        
        if label_support in {True, False}:
            log.warning('label_support argument is deprecated. Use mask instead.')
            #raise ValueError()

        self.mask = mask

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        import third_party.PFENet.util.transform as transform

        if mode == 'val':
            data_list = expanduser('~/projects/old_one_shot/PFENet/lists/pascal/val.txt')

            data_transform = [transform.test_Resize(size=image_size)] if image_size != 'original' else []
            data_transform += [
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)
            ]   


        elif mode == 'train':
            data_list =  expanduser('~/projects/old_one_shot/PFENet/lists/pascal/voc_sbd_merge_noduplicate.txt')

            assert image_size != 'original'

            data_transform = [
                transform.RandScale([0.9, 1.1]),
                transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
                transform.RandomGaussianBlur(),
                transform.RandomHorizontalFlip(),
                transform.Crop((image_size, image_size), crop_type='rand', padding=mean, ignore_label=255),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)
            ]

        data_transform = transform.Compose(data_transform)

        self.dataset = SemData(split=split, mode=mode, data_root=expanduser('~/datasets/PascalVOC2012/VOC2012'), 
                               data_list=data_list, shot=1, transform=data_transform, use_coco=False, use_split_coco=False)

        self.class_list = self.dataset.sub_val_list if mode == 'val' else self.dataset.sub_list

        # verify that subcls_list always has length 1
        # assert len(set([len(d[4]) for d in self.dataset])) == 1

        print('actual length', len(self.dataset.data_list))

    def __len__(self):
        if self.mode == 'val':
            return len(self.dataset.data_list)
        else:
            return len(self.dataset.data_list)

    def __getitem__(self, index):
        if self.dataset.mode == 'train':
            image, label, s_x, s_y, subcls_list = self.dataset[index % len(self.dataset.data_list)]
        elif self.dataset.mode == 'val':
            image, label, s_x, s_y, subcls_list, ori_label = self.dataset[index % len(self.dataset.data_list)]
            ori_label = torch.from_numpy(ori_label).unsqueeze(0)
            
            if self.image_size != 'original':
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda()*255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                label = backmask.clone().long()      
            else:
                label = label.unsqueeze(0) 

            # assert label.shape == (473, 473)

        if self.p_negative > 0:
            if torch.rand(1).item() < self.p_negative:
                while True:
                    idx = torch.randint(0, len(self.dataset.data_list), (1,)).item()
                    _, _, s_x, s_y, subcls_list_tmp, _ = self.dataset[idx]
                    if subcls_list[0] != subcls_list_tmp[0]:
                        break

        s_x = s_x[0]
        s_y = (s_y == 1)[0]
        label_fg = (label == 1).float()
        val_mask = (label != 255).float()

        class_id = self.class_list[subcls_list[0]]

        label_name = PASCAL_CLASSES[class_id][0]
        label_add = ()
        mask = self.mask

        if mask == 'text':
            support = ('a photo of a ' + label_name + '.',)
        elif mask == 'separate':
            support = (s_x, s_y)
        else:
            if mask.startswith('text_and_'):
                label_add = (label_name,)
                mask = mask[9:]

            support = (blend_image_segmentation(s_x, s_y.float(), mask)[0],)

        return (image,) + label_add + support, (label_fg.unsqueeze(0), val_mask.unsqueeze(0), subcls_list[0])        
