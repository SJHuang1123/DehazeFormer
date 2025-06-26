import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
from torchvision import transforms

t = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
])

class HyperspectralDehazeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.gt_dir = os.path.join(root_dir, 'GT')
        self.trans_dir = os.path.join(root_dir, 'trans')
        self.gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.mat')])
        self.transform = transform if transform is not None else t

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        trans_path = os.path.join(self.trans_dir, self.gt_files[idx])

        # Load .mat files
        gt_mat = loadmat(gt_path)
        trans_mat = loadmat(trans_path)

        # Automatically find the data key (assumes one main variable)
        gt = self._extract_data(gt_mat)
        trans = self._extract_data(trans_mat)

            
        # Ensure data is float32 and shaped as (C, H, W)
        gt = self._normalize_and_reshape(gt)
        trans = self._normalize_and_reshape(trans)
        
        if self.transform:
            gt = self.transform(gt)
            trans = self.transform(trans)

        sample = {'trans': trans, 'gt': gt}


        return sample

    def _extract_data(self, mat_dict):
        return mat_dict['gt'] if 'gt' in mat_dict else mat_dict['t_new']

    def _normalize_and_reshape(self, array):
        array = array.astype(np.float32)
        if array.ndim == 3 and array.shape[2] == 172:
            # H x W x C -> C x H x W
            array = np.transpose(array, (2, 0, 1))
        elif array.ndim == 4 and array.shape[0] == 1:
            # 1 x C x H x W -> C x H x W
            array = array[0]
        else:
            raise ValueError(f"Unsupported shape {array.shape}, expected (256,256,172) or (1,172,256,256)")
        return torch.from_numpy(array)

if __name__ == "__main__":
    dataset = HyperspectralDehazeDataset(root_dir='/home/q36131207/HSID_dataset/AVIRIS/qtrain')
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(sample['gt'].size(), sample['trans'].size())