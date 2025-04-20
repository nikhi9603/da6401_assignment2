import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import os


def validationDataSplit(train_dataset):
  classLabels = [label for _,label in train_dataset.samples]
  num_classes = len(np.unique(classLabels))

  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  train_indices, val_indices = next(sss.split(train_dataset.samples, classLabels))

  train_subset = Subset(train_dataset, train_indices)
  val_subset = Subset(train_dataset, val_indices)
  return train_subset, val_subset, num_classes


def load_data(base_dir, isDataAug, batch_size):
  train_dir = os.path.join(base_dir, 'train')
  test_dir = os.path.join(base_dir, 'val')

  train_transform, test_transform = None, None

  if isDataAug == False:
    train_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  else:
    train_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
  test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
  train_dataset, val_dataset, num_classes = validationDataSplit(train_dataset)

  # print(f"inp: {train_dataset[0][0].shape} {train_dataset[0][1]}")

  train_loader = DataLoader(train_dataset,shuffle=True,num_workers=2,batch_size=batch_size,pin_memory=True)
  test_loader = DataLoader(test_dataset,shuffle=True,num_workers=2,batch_size=64,pin_memory=True)
  val_loader = DataLoader(val_dataset,shuffle=True,num_workers=2,batch_size=64,pin_memory=True)

  return train_loader, test_loader, val_loader, num_classes

# load_data("/content/drive/MyDrive/DL_Assignment2/Dataset/inaturalist_12K/", True)
