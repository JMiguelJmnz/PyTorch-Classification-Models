### Image Classification Using PyTorch

### Project Overview
This project involves building a deep learning model to perform image classification using PyTorch framework.
<br>The goal is to develop a neural network capable of automatically classifying images of animals into three predefined categories: dog, cat, and wild, based on their content.

### Importing libraries
For this project, this are the used libraries

```python
import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import os
```

As well as loading the dataset
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/andrewmvd/animal-faces")
```

### Selecting device for computation
In this step we select (if available) our CUDA GPU to speed up the training of the model
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available: ", device)
print("Is Cuda available?: ", torch.cuda.is_available()) 
print(torch.cuda.get_device_name())
```

### Setting up the dataset
Now we load the data to train the model and create a new dataframe with the filepath and label of each item
```python
base_path = Path(os.getcwd()).resolve()

image_path = []
labels = []

afhq_path = base_path / "animal-faces" / "afhq"

for i in os.listdir(afhq_path):
    label_path = afhq_path / i
    for label in os.listdir(label_path):
        image_path_label_path = label_path / label
        for image in os.listdir(image_path_label_path):
            image_path.append(image_path_label_path / image)
            labels.append(label)

data_df = pd.DataFrame(zip(image_path, labels), columns=["image_path", "labels"])

print(data_df["labels"].unique())
data_df.head()
```
![image](https://github.com/user-attachments/assets/7e5c472d-5d06-494f-8430-4af687ebb909)

# Train, test and validation groups
```python
train = data_df.sample(frac = 0.7)
test = data_df.drop(train.index)

val = test.sample(frac = 0.5)
test = test.drop(val.index)
```
After creating the groups we proceed with to transform the information so we can use it

by using label_encoder we'ld change the labels of the images to numerical values so our code can interpret them
```python
label_encoder = LabelEncoder()
label_encoder.fit(data_df["labels"])
```

With transform we will be able to addapt each image to the required size and format
```python
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])
```

Now we create the class to help us process all the data for the training step
```python
class CustomImageDataset(Dataset):
    def __init__(self,dataframe,transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe['labels'])).to(device)

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image).to(device)
            
        return image, label
```
And we use it for our training, validation and test groups

```python
train_dataset = CustomImageDataset(dataframe = train, transform = transform)
val_dataset = CustomImageDataset(dataframe = val, transform = transform)
test_dataset = CustomImageDataset(dataframe = test, transform = transform)
```
