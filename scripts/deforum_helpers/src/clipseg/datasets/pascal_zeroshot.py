from os.path import expanduser
import torch
import json
import torchvision
from general_utils import get_from_repository
from general_utils import log
from torchvision import transforms

PASCAL_VOC_CLASSES_ZS = [['cattle.n.01', 'motorcycle.n.01'], ['aeroplane.n.01', 'sofa.n.01'], 
                         ['cat.n.01', 'television.n.03'], ['train.n.01', 'bottle.n.01'],
                          ['chair.n.01', 'pot_plant.n.01']]


class PascalZeroShot(object):

    def __init__(self, split, n_unseen, image_size=224) -> None:
        super().__init__()

        import sys
        sys.path.append('third_party/JoEm')
        from third_party.JoEm.data_loader.dataset import VOCSegmentation
        from third_party.JoEm.data_loader import get_seen_idx, get_unseen_idx, VOC

        self.pascal_classes = VOC
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

        if split == 'train':
            self.voc = VOCSegmentation(get_unseen_idx(n_unseen), get_seen_idx(n_unseen), 
                                       split=split, transform=True, transform_args=dict(base_size=312, crop_size=312), 
                                       ignore_bg=False, ignore_unseen=False, remv_unseen_img=True)
        elif split == 'val':
            self.voc = VOCSegmentation(get_unseen_idx(n_unseen), get_seen_idx(n_unseen), 
                                       split=split, transform=False, 
                                       ignore_bg=False, ignore_unseen=False)

        self.unseen_idx = get_unseen_idx(n_unseen)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, i):

        sample = self.voc[i]
        label = sample['label'].long()
        all_labels = [l for l in torch.where(torch.bincount(label.flatten())>0)[0].numpy().tolist() if l != 255]
        class_indices = [l for l in all_labels]
        class_names = [self.pascal_classes[l] for l in all_labels]

        image = self.transform(sample['image'])

        label = transforms.Resize((self.image_size, self.image_size), 
            interpolation=torchvision.transforms.InterpolationMode.NEAREST)(label.unsqueeze(0))[0]

        return (image,), (label, )


