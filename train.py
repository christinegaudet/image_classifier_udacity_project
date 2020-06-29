import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import torch
import seaborn as sns
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse



# defines parameters from command line and turns them into variables to use in
def define_parameters():
  parser = argparse.ArgumentParser(
    description='Deep learning prediction model - training script')

  # select data directory
  parser.add_argument(
    'data_directory',
    type=str,
    help='The data directory to conduct the analysis')

  # save directory option
  parser.add_argument('--sav_directory', help='To save the model')

  # model architect
  parser.add_argument(
    '--arch',
    help='Select model architecture',
     default='vgg16')

  # hyperparameter -> learning rate
  parser.add_argument('--learning_rate', help='Defining learning rate hyperparameter', default=0.001)

  # hyperparameters -> hidden_units
  parser.add_argument(
    '--hidden_units',
    help='Defining hyperparameters',
     default=512)

  # hyperparameters -> epochs
  parser.add_argument(
    '--epochs',
    help='Defining epoch hyperparameters',
     default=20)

  # gpu option
  parser.add_argument(
    '--gpu',
    type=bool,
     help='Turning on the GPU to run the model')

  args = parser.parse_args()
  return args


def validation(model, valid_dataloader, criterion):
    valid_loss = 0
    accuracy = 0

    for ii, (images, labels) in enumerate(valid_dataloader):

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)

        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)

        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy


def transform(data_dir, valid_dir, test_dir):
  data_train_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

  data_valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

  image_train_datasets = datasets.ImageFolder(
    train_dir, transform=data_train_transforms)
  image_valid_datasets = datasets.ImageFolder(
    train_dir, transform=data_valid_transforms)
  image_test_datasets = datasets.ImageFolder(
    train_dir, transform=data_valid_transforms)

  train_dataloader = torch.utils.data.DataLoader(
    image_train_datasets, batch_size=64, shuffle=True)
  valid_dataloader = torch.utils.data.DataLoader(
      image_valid_datasets, batch_size=64)
  test_dataloader = torch.utils.data.DataLoader(
      image_test_datasets, batch_size=64)
  return image_train_datasets, image_valid_datasets, image_test_datasets, train_dataloader, valid_dataloader, test_dataloader


def train(epochs, train_dataloader):
  running_loss = 0
  steps = 0 
  print_every = 20
  for e in range(epochs):
    since = time.time()
    for ii, (images, labels) in enumerate(valid_dataloader):
      steps += 1

      images, labels = images.to('cuda'), labels.to('cuda')
      optimizer.zero_grad()

      # Forward pass and backward passe
      outputs = model.forward(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Calculating running loss
      running_loss += loss.item()

      if steps % print_every == 0:
        model.eval()

      with torch.no_grad():
        valid_loss, accuracy = validation(model, valid_dataloader, criterion)

      print(f"Epoch: {e+1}, \
              Training Loss: {round(running_loss/print_every,3)} \
              Validation Loss: {round(valid_loss/len(valid_dataloader),3)} \
              Validation Accuracy: {round(float(accuracy/len(valid_dataloader)),3)}")
            
      running_loss = 0
            
      model.train()
            
      time_taken = time.time() - since
  print(f"Time taken for epoch: {time_taken} seconds")
  return model 

if __name__ == "__main__":

  # getting arguments from the parameters functions 
  input = define_parameters()

  data_directory = input.data_directory
  sav_directory = input.sav_directory
  arch = input.arch 
  learning_rate = input.learning_rate
  hidden_units = input.hidden_units
  epochs = input.epochs 
  gpu = input.gpu 

  # loading the data 
  data_dir = data_directory 
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'

  # transforming data 
  image_train_datasets, image_valid_datasets, image_test_datasets, train_dataloader, valid_dataloader, test_dataloader = transform(data_dir, valid_dir, test_dir)
  

  # label mapping 
  with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    df = pd.DataFrame({'name': cat_to_name})
    print("Sample of data:")
  print(df.head())

  # building and transforming the classifier 
  exec("model = models.{}(pretrained=True)".format(arch))
  print(model)

  for params in model.parameters():
    params.requires_grad = False 

  D_in, H, D_out = 25088, 4096, 102

  classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(D_in, H)),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(H,D_out)),
                          ('ouput', nn.LogSoftmax(dim=1))
                          ]))

  model.classifier = classifier 

  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # training the model
  model = train(epochs, train_dataloader)
  print("\nThe model has completed training!")


  # testing your network
  model.eval()
  model = validation(model, test_dataloader, criterion)
  accuracy = equal/len(test_dataloader)*100
  print("Test score accuracy results is {:.3f} %".format(accuracy))

  # save the checkpoint
  model.class_to_idx = image_train_datasets.class_to_idx

  checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': image_train_datasets.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs}

  torch.save(checkpoint, 'checkpoint.pth')








