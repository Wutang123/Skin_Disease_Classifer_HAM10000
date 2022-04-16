#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      dataset.py
# Create:        04/16/22
# Description:   Function used for dataset loading
#---------------------------------------------------------------------

# IMPORTS:
# Python libraries
from PIL import Image

# pytorch libraries
import torch
from torch.utils.data import Dataset

# Class:
#---------------------------------------------------------------------
# Function:    Dataset()
# Description: Characterizes HAM10000 for PyTorch
#---------------------------------------------------------------------
class dataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, df, transform = None):
        'Initialization'
        self.df = df
        self.transform = transform

    def __len__(self):
        # Denotes the total number of samples
        return len(self.df)

    def __getitem__(self, index):
        # Generates one sample of data
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y