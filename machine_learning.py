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
import random

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

train_data = Init_selection(train_list, transform = train_transforms)
test_data = Init_selection (test_list, transform = test_transforms)
val_data = Init_selection(val_list, transform = test_transforms)
# print(len(train_data))
# print(len(test_data))
# print(len(val_data))
# print(train_list[0])

train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True )
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = True)

# print(len(train_loader))
# print(len(test_loader))
# print(len(val_loader))

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc1 = nn.Linear(3*3*64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = Cnn().to(device)
print(model.train())

optimizer = optim.Adam(params = model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(
        epoch+1, epoch_accuracy, epoch_loss))

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch + 1, epoch_val_accuracy, epoch_val_loss))

probs = []
model.eval()
with torch.no_grad():
    for data, fileid in test_loader:
        data = data.to(device)
        preds = model(data)
        preds_list = F.softmax(preds, dim = 1)[:, 1].tolist()
        probs += list(zip(list(fileid), preds_list))

probs.sort(key = lambda x: int(x[0]))
print(probs)

idx = list(map(lambda x: x[0], probs))
prob = list(map(lambda x: x[1], probs))

submission = pd.DataFrame({'id':idx,'label':prob})

print(submission)

submission.to_csv('result.csv', index = False)

id_list = []
class_ = {0: 'rose', 1: 'tulip'}

fig, axes = plt.subplots(2, 5, figsize = (20, 12), facecolor = 'w')

for ax in axes.ravel():

    i = random.choice(submission['id'].values)

    label = submission.loc[submission['id'] == i, 'label'].values[0]
    if label > 0.5:
        label = 1
    else:
        label = 0

    img_path = os.path.join(test_list, '{}.jpg'.format(i))
    img = Image.open(img_path)

    ax.set_title(class_[label])
    ax.imshow(img)
