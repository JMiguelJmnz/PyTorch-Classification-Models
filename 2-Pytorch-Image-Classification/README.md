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
Stablishing of the Learning rate, barch size and epochs to be used:
```python
LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 10
```
Now we create DataLoaders for the training, validation and testing using the batch size defined before and using shuffle to randomize the order and get a better generalization during training.  
For the test set, shuffle is set to False to maintain a consistent evaluation.
```python
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)
```
Set up the CNN where we'll start with images as the imput with 128x128 size, we take 3 chanels as it is RGB and output 32 features, on the next layer we'll double the output to 64 features and a third layer with 128 features
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.convl = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(2,2)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128*16*16), 128)

        self.output = nn.Linear(128, len(data_df['labels'].unique()))

    def forward(self, x):
        x = self.convl(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x
```
Here we create an instance of the class we created and allocate it to the GPU 
```python
model = Net().to(device)
```
Set up the criterion used to calculate the loss using CrossEntropyLoss and set up the optimizer using Adam
```python
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = LR)
```
Now to train the model, the lists for plotting are initialized to be used at the end.
```python
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []
```
Then we set at the start of each epoch the variables to accumulate the total values for each of the metrics and reset them to zero followed by the training and validation phases storing the results of the metrics at the end
```python
for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0
    total_loss_val = 0
    total_acc_val = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)    

        labels = labels.long()
        optimizer.zero_grad()

        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        total_loss_train += train_loss.item()

        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()

        total_acc_train += train_acc
        optimizer.step()
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.long()
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            total_loss_val += val_loss.item()

            val_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
            total_acc_val += val_acc
    
    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_validation_plot.append(round(total_loss_val/1000, 4))
    
    total_acc_train_plot.append(round((total_acc_train/ train_dataset.__len__()) * 100, 4))
    total_acc_validation_plot.append(round((total_acc_val/ val_dataset.__len__()) * 100, 4))

    print(f'''Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/1000, 4)}  Train Accuracy {round((total_acc_train/ train_dataset.__len__()) * 100, 4)}
            Validation Loss {round(total_loss_val/1000, 4)} Validation Accuracy {round((total_acc_val/ val_dataset.__len__()) * 100, 4)}
        ''')
```
For this case, this were the training statistics
![image](https://github.com/user-attachments/assets/023cbcfc-8e0d-488d-824d-a9b0ab65bf3e)

Now we test the accuracy and loss of the model
```python
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.long()
        predictions = model(inputs)

        acc = (torch.argmax(predictions, axis = 1) == labels).sum().item()
        total_acc_test += acc
        test_loss = criterion(predictions, labels)
        total_loss_test += test_loss.item()

    print(f'Accuracy Score is: {round(total_acc_test/test_dataset.__len__() * 100, 4)} and Loss is {round(total_loss_test/1000, 4)}')
```
Getting in this case an Accuracy Score 96.2381 and Loss of 0.0172

We can visualize the model's performance with the plotting data we kept track of
```python
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_validation_plot, label='Validation Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
axs[1].set_title('Training and Validation Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.show()
```

Left Plot (Loss):
Shows the training and validation loss at each epoch. This helps track how well the model is minimizing the error function over time.
A decreasing training loss indicates that the model is learning.
If the validation loss stops decreasing or increases, it may suggest overfitting.

Right Plot (Accuracy):
Displays the model’s accuracy on both the training and validation sets per epoch.
An increasing trend in both means performance is improving.
A gap between training and validation accuracy may indicate overfitting or underfitting.
d![image](https://github.com/user-attachments/assets/9af3887e-2b8d-4446-8c99-9b498880f895)

Lastly a function is created to make predictions on new images where it reads the image, transforms it using transform object, make the prediction and inverse transform by label encoder to output the prediction
```python
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).to(device)

    output = model(image.unsqueeze(0))

    output = torch.argmax(output, axis=1).item()
    return label_encoder.inverse_transform([output])
```

### Testing with a new input
![AnimalSample](https://github.com/user-attachments/assets/773a9fcc-7fd0-439d-8008-bc6b3fd9e5e8)

predict_image("AnimalSample.jpg")
