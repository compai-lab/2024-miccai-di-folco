import torch
from torch.utils.data import (Dataset, DataLoader)
import torchvision.transforms as transforms
import pytorch_lightning as pl
from monai.transforms import (AddChannel, Compose, RandRotate, RandZoom,
    Resize, RandShiftIntensity, ToTensor, RandFlip, AdjustContrast, ScaleIntensity)
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
import h5py
import json
import os
import torchvision.transforms as transforms


class CardiacLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()


        self.batch_size = args['batch_size'] if 'batch_size' in args.keys() else 16
        self.attributes_idx = args['attributes_idx']
        self.folder = args['attributes_path']
        self.patch_file = args['patch_path']
        self.attributes_path = args['attributes_path'] + f'full_info.csv'

        self.win_size = args['win_size']
        self.gamma = args['rescale']
        self.binary_label = args['binary_label'] if 'binary_label' in args.keys() else False

        self.moment = args['moment'] if 'moment' in args.keys() else None

        patch_name = os.path.split(self.patch_file)[1]
        patch_root = os.path.split(self.patch_file)[0]
        testing_path = patch_root + '/testing_' + patch_name
        if Path(testing_path).is_file() and Path(args['attributes_path'] + f'testing_full_info.csv').is_file():

            self.testing_patch_file = testing_path
            self.testing_attributes_path = args['attributes_path'] + f'testing_full_info.csv'

        self.train_transforms = Compose(
            [
                ToTensor(),
                AddChannel(),
                transforms.CenterCrop(self.win_size[0]),
                #RandZoom(prob=0.1, min_zoom=0.9, max_zoom=1.1),
                AdjustContrast(gamma = self.gamma),
                ScaleIntensity(minv=0.0, maxv=1.0)
            ]
        )
        self.val_transforms = Compose(
            [
                ToTensor(),
                AddChannel(),
                transforms.CenterCrop(self.win_size[0]),
                AdjustContrast(gamma=self.gamma),
                ScaleIntensity(minv=0.0, maxv=1.0)
            ]
        )
        #self.val_transforms = self.train_transforms
        self.setup()

    def setup(self):

        dataset = CardiacCustomDataset(self.patch_file, self.attributes_idx, self.attributes_path,
                                       self.train_transforms, self.binary_label, self.moment)
        self.attributes_dict = dataset.attributes_dict

        try:
            with open(self.folder + 'split.json', 'r') as json_file:
                data = json.load(json_file)
            
            indices_train = data['indices_train']
            indices_val = data['indices_val']
            indices_test = data['indices_test']

        except:
            labels = dataset.csv_attributes['label']
            indices_train, indices_val = train_test_split(list(range(len(dataset))),
                                                      stratify=labels, test_size=0.15)

        self.train_set = torch.utils.data.Subset(dataset, indices_train)
        self.val_set = torch.utils.data.Subset(dataset, indices_val)

        if not 'indices_test' in locals():
            indices = indices_val
            val_dataset = self.val_set
            indices_val, indices_test = train_test_split(indices, test_size=0.5)
            self.val_set = torch.utils.data.Subset(dataset, indices_val)
        self.test_set = torch.utils.data.Subset(dataset, indices_test)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False)

    def get_attributes_dict(self):
        return self.attributes_dict

    def get_attributes_idx(self):
        return self.attributes_idx


class CardiacCustomDataset(Dataset):
    def __init__(self, patch_file, attributes_idx, attributes_path, transforms=None, binary_label=False, moment= None):
        super().__init__()

        self.patch_file = h5py.File(patch_file, 'r')
        self.attributes_idx = attributes_idx
        self.csv_attributes = pd.read_csv(attributes_path,  dtype={'pid':str})

        tmp_dict = self.csv_attributes.drop(['label','pid'], axis=1)
        self.attributes_dict = list(tmp_dict.columns)
        self.binary_label = binary_label
        self.transform = transforms

        K = list(self.patch_file.keys())[0]

        self.moment = moment


    def get_attributes(self, idx):

        attributes = []

        for attr in self.attributes_idx:
            try:
                label = self.csv_attributes.loc[idx][attr]
            except:
                print(attr)
                raise f'{attr} not existing in attributes csv file'
            attributes.append(label)

        full_attributes = self.csv_attributes.loc[idx, self.attributes_dict].values
        return torch.Tensor(attributes), torch.Tensor(full_attributes.tolist())

    def get_case(self,idx):
        #with h5py.File(self.patch_file, 'r') as ff:
        if self.moment is None:
            data = self.patch_file[f'{idx}'][:]
        else:
            data = {'ED':self.patch_file[f'{idx}/ED'][:], 'ES': self.patch_file[f'{idx}/ES'][:]}
        return data

    def __len__(self):
        return len(self.csv_attributes['pid'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.csv_attributes.loc[idx]['label']
        if self.binary_label and label > 1:
            label = 1
        idx_name = self.csv_attributes.loc[idx]['pid']

        patch = self.get_case(idx_name)
        attributes, full_attributes = self.get_attributes(idx)

        if self.moment is None:
            img = self.transform(patch)
        elif self.moment == 'ED':
            img = self.transform(patch['ED'])
        elif self.moment == 'ES':
            img = self.transform(patch['ES'])
        elif self.moment == 'all':
            img_ed = self.transform(patch['ED'])
            img_es = self.transform(patch['ES'])
            img = np.concatenate((img_ed, img_es), axis = 0)

        return img, label, attributes, full_attributes