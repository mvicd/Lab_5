import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

lr = 0.001 # learning_rate
batch_size = 100 # we will use mini-batch method
epochs = 10 # How much to train a model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)
if device == 'cuda':
    torch.cuda.manual_seed_all(1234)

directory =  "D:\Lab Python\dataset_3"
# test_dir = "D:\Lab Python\Lab_1\dataset\ rose"

directory_list = glob.glob((os.path.join(directory, "*.jpeg")))

print(len(directory_list))

train_list, test_list = train_test_split(directory_list, test_size = 0.1)
train_list, val_list = train_test_split(train_list, test_size = 0.1111)

print(len(train_list))
print(len(test_list))
print(len(val_list))

# print(len(train_directory) / len(directory_list))
# print(len(test_directory) / len(directory_list))
# print(len(val_directory) / len(directory_list))

# random_idx = np.random.randint(1, len(train_list), size = 10)
# fig = plt.figure()
# i = 1

# for idx in random_idx:
#     ax = fig.add_subplot(2, 5, i)
#     img = Image.open(train_list[idx])
#     plt.imshow(img)
#     i += 1
#     plt.axis('off')
#     plt.tick_params(axis = 'both', left = False, top = False, right = False, bottom = False, labelleft = False, labeltop = False, labelright = False, labelbottom = False)
#     plt.savefig('foo.png', dpi = 100, bbox_inches = 'tight', pad_inches = 0.0)

# plt.show()

print(train_list[0].split('\\')[-1].split('.')[0])
train_list[0].split('_')[-1].split('.')[0]

print(int(train_list[0].split('.')[1]))

train_transforms =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
     transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

class dataset(Dataset):
    def __init__(self, file_list, transform = None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)

        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('/')[-1].split('.')[0]

        if label == 'rose':
            label = 0
        elif label == 'tulip':
            label = 1

        return img_transformed, label
