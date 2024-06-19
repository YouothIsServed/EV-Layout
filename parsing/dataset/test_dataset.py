import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json
import copy
from PIL import Image
from skimage import io
import os
import os.path as osp
import numpy as np
class TestDatasetWithAnnotations(Dataset):
    '''
    Format of the annotation file
    annotations[i] has the following dict items:
    - filename  # of the input image, str
    - height    # of the input image, int
    - width     # of the input image, int
    - lines     # of the input image, list of list, N*4
    - junctions # of the input image, list of list, M*2
    '''

    def __init__(self, root, ann_file, transform = None):
        self.root = root
        with open(ann_file, 'r') as _:
            self.annotations = json.load(_)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        # image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        vg = np.load(self.root + '/' + ann['filename'].split('.')[0] + '.npy')
        vg = vg.transpose((2, 0, 1))
        vg = vg[3:, :, :]
        # vg = vg[[1, 3], :, :]
        image = vg
        for key,_type in (['junctions',np.float32],
                          ['junctions_semantic',np.int64],
                          ['edges_positive',np.int64],
                          ['edges_negative',np.int64],
                          ['edges_semantic',np.int64]):

            ann[key] = np.array(ann[key],dtype=_type)

        if not 'junctions_semantic' in ann:
            ann['junc_occluded'] = np.array(ann['junc_occluded'],dtype=np.bool)

        assert np.all(ann['edges_semantic'] > 0) #Assumes that the dataset also has an invalid class as 0

        map_path = self.root
        map_oirg = np.load(map_path + '/orig_map/' + ann['filename'].split('.')[0] + '.npy').astype(np.float32)
        map_shift = np.load(map_path + '/shift_map/' + ann['filename'].split('.')[0] + '.npy').astype(np.float32)
        ann['map_orig'] = torch.tensor(map_oirg)
        ann['map_shift'] = torch.tensor(map_shift)

        if self.transform is not None:
            waste_image = image.transpose((1, 2, 0))
            waste_image, ann = self.transform(waste_image,ann)
        image = torch.tensor(image).float()
        return image, ann



    def image(self, idx):
        ann = self.annotations[idx]
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        vg = np.load(self.root +'/'+ ann['filename'].split('.')[0] + '.npy')
        vg = vg.transpose((2, 0, 1))
        vg = vg[3:, :, :]
        # vg = vg[[1, 3], :, :]
        image = vg
        return image
    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])
