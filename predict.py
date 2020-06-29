import torch
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
import matplotlib.pyplot as plt 
import PIL
from PIL import Image


# defines parameters from command line and turns them into variables to use in
def define_parameters():
  parser = argparse.ArgumentParser(
    description='Deep learning prediction model - prediction script')

  # select the path of the image
  parser.add_argument(
    'image_path',
    type=str,
    help='Enter the file path of the testing image')

  # select the path of the image
  parser.add_argument(
    '--topk',
    type=int,
    help='Enter the number of top classes to analyze', default=5)

  # gpu option
  parser.add_argument(
    '--gpu',
    type=bool,
     help='Turning on the GPU to run the model')

  args = parser.parse_args()
  return args

# loading the model 
def loading_checkpoint(filepath):
  checkpoint = torch.load(filepath, map_location={'cuda:0': 'cpu'})
    
  model.load_state_dict(checkpoint['state_dict'])
  model.classifier = checkpoint['classifier']
  model.class_to_idx = checkpoint['class_to_idx']
    
  return model

 # scales, crops, and normalizes a PIL image for a PyTorh model, returns an Numpy array
def process_image(image_path):       
  #opening image
  image = PIL.Image.open(image_path)

  #resizing     
  width, height = image.size
  if width > height:
    r = width/height
    size = int(256 * r),256
  elif height > width:
    r = height/width
    size = 256,int(256 * r)
  else:
    size = 256,256
    image = image.resize(size)
    
  cropped_image = image.crop((size[0]//2-112,size[1]//2-112,size[0]//2+112,size[1]//2+112))
    
  #convert color channel to floats between 0 & 1 
  img_array = np.array(cropped_image)
  np_image = img_array/255
    
  # normalize color channels
  means = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  np_image = (np_image-means)/std
        
  # set the color to the first channel
  np_image = np_image.transpose(2, 0, 1)
  
  return np_image

# to check work 
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# function to make predictions 
def predict(image_path, model, topk=5):
    
  model.to("cpu")
  image = process_image(image_path)
  torch_image = torch.from_numpy(np.expand_dims(image, axis=0)).type(torch.FloatTensor).to("cpu")
  model.eval()
  output = model(torch_image)
  pr_output = torch.exp(output)
  probs,indices = pr_output.topk(topk)
  probs = np.array(probs.detach())[0]
  indices = np.array(indices.detach())[0]

  # convert indices to actual category names
  index_to_class = {val: key for key, val in model.class_to_idx.items()}
  top_classes = [index_to_class[each] for each in indices]

  return probs,top_classes


if __name__ == "__main__":

  # getting arguments from the parameters functions 
  input = define_parameters()

  image_path = input.image_path
  topk_value = input.topk
  gpu = input.gpu 

  model = loading_checkpoint('checkpoint.pth')
  print(model)

  # testing code 
  imshow(process_image(image_path)) 

  # make predictions
  predictions = predict(image_path, model, topk=topk_value)
  print("Top 2 Most Probable matches:")
  print(predictions)

  predictions = predict(image_path, model, topk=topk_value)
  print("Top 5 Most Probable matches")
  print(predictions)

  # converting results above into actual names  
  img = process_image(image_path)
  imshow(img)
  plt.show()
  probs, classes = predict (image_path, model, topk_value)

  class_names = [cat_to_name [item] for item in classes]
 
  plt.figure(figsize = (6,10))
  plt.subplot(2,1,2)

  sns.barplot(x=probs, y=class_names, color= 'green');

  plt.show()  






  