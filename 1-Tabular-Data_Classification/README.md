### Tabular Classification using Pytorch

### Project Overview  
This project was created as a hands-on exercise to learn and apply PyTorch for tabular data classification, the goal was to build and train a deep learning model using PyTorch to classify rice grain types from a tabular dataset.

### Importing libraries
For this project, theses are the libraries used
```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
```
As well as loading the dataset
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification")
```
### Selecting device for computation
In this step we select (if available) our CUDA GPU to speed up the training of the model
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name())  # Should print your GPU name
```
### Setting up the data  
```python
base_path = Path(os.getcwd()).resolve()

data_df = pd.read_csv(base_path/"rice-type-classification"/"riceClassification.csv")
data_df.head()
```
![image](https://github.com/user-attachments/assets/c3442963-c634-44b8-90b0-25f6c76312b8)
Here we do a quick clean of the data and drop the columns we don't need.
```python
data_df.dropna(inplace=True)
data_df.drop(['id'], axis = 1, inplace = True)
```
What the model is going to determine is the Class of the rice, we can see how many classes there are with
```python
print(data_df["Class"].unique())
```
[1 0]
Where 1 stands for Jasmine and 0 stands for Gonen.

For the data processing we normalize the information so every feature contribute equally to the learning process
```python
original_df = data_df.copy()

for column in data_df.columns:
    data_df[column] = data_df[column]/data_df[column].abs().max()
```
At this point we can Separate our dataframe into our features and the classes
```python
X = np.array(data_df.iloc[:,:-1])
y = np.array(data_df.iloc[:,-1])
```
With this step done, the training, test and validation groups can be created
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
```
Now we can create our class that will help our datasets to be structure for use with DataLoader
```python
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
```
Wrap the groups in the TabularDataset class
```python
training_data = TabularDataset(X_train, y_train)
validation_data = TabularDataset(X_val, y_val)
testing_data = TabularDataset(X_test, y_test)
```
Pass the datasets into DataLoader objects
```python
train_dataloader = DataLoader(training_data, batch_size = 32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size = 32, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size = 32, shuffle=False)
```
Definining a feedforward neural network for binary classification
```python
HIDDEN_NEURONS = 10

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.linear = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
model = MyModel().to(device)
```
Set up the criterion used to calculate the loss using BinaryCrossEntropyLoss and set up the optimizer using Adam
```python
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr = 1e-3)
```
Now we create the training and evaluation loop that runs for a specified number of epochs, updating the model's parameters and tracking its progress through both training and validation datasets.
```python
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

epochs = 10
for epoch in range(epochs):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    for data in train_dataloader:
        inputs, labels = data
        
        prediction = model(inputs).squeeze(1)

        batch_loss = criterion(prediction, labels)

        total_loss_train += batch_loss.item()

        acc = ((prediction).round() == labels).sum().item()

        total_acc_train += acc

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data

            prediction = model(inputs).squeeze(1)
            batch_loss = criterion(prediction, labels)

            total_loss_val += batch_loss.item()
            acc = ((prediction).round() == labels).sum().item()

            total_acc_val += acc
    total_loss_train_plot.append(round(total_loss_train/1000, 4))  
    total_loss_validation_plot.append(round(total_loss_val/1000, 4))  

    total_acc_train_plot.append(round(total_acc_train/training_data.__len__() * 100, 4))  
    total_acc_validation_plot.append(round(total_acc_val/validation_data.__len__() * 100, 4))  

    print(f'''Epoch no. {epoch+1} Train Loss:{round(total_loss_train/1000, 4)} Train Accuracy {round(total_acc_train/training_data.__len__() * 100, 4)}
            Validation Loss: {round(total_loss_val/1000, 4)} Validation Accuracy: {round(total_acc_val/validation_data.__len__() * 100, 4)}''')
    print("="*25)
```    
Results of the training:
![image](https://github.com/user-attachments/assets/f4add610-49a8-4c1c-8c33-984fdfca2c00)

Then we run the testing:
```python
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    
    for data in testing_dataloader:
        input, labels = data
        
        prediction = model(input).squeeze(1)

        batch_loss_test = criterion(prediction, labels).item()
        total_loss_test += batch_loss_test

        acc = ((prediction).round() == labels).sum().item()

        total_acc_test += acc

print("Accuracy: ", round(total_acc_test/testing_data.__len__() *100, 4))
```
Accuracy:  98.5704

To plot the loss and accuracy we simply use the values set in the training loop
```python
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(15,5))

axs[0].plot(total_loss_train_plot, label = 'Training Loss')
axs[0].plot(total_loss_validation_plot, label = 'Validation Loss')
axs[0].set_title("Training and Validation loss over epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].set_ylim([0,2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label = 'Training Accuracy')
axs[1].plot(total_acc_validation_plot, label = 'Validation Accuracy')
axs[1].set_title("Training and Validation Accuracy over epochs")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].set_ylim([0,100])
axs[1].legend()

plt.show()
```
![image](https://github.com/user-attachments/assets/27d80501-26c0-4dbe-80de-66bab5539d31)
with these results we can see that the model is underfiting due to the quick and small stabilization of the training curves, this could suggest thtat the model is not learning effectively do to being to simple.  

By tweaking some parameters like lowering the learning rate to 1e-4 and adjusting the hidden layers to 15, we can see it improves the model although it could be refined a lot more.
![image](https://github.com/user-attachments/assets/d03fd846-3b8d-4c88-ae2c-e7881f24bd3d)

Whit these results we can try the model with a new sample inputing the new values for all the features and normalizing them for the model to process
```python
Area = 2353/original_df['Area'].abs().max()
MajorAxisLength = 42/original_df['MajorAxisLength'].abs().max()
MinorAxisLength = 81/original_df['MinorAxisLength'].abs().max()
Eccentricity = 12/original_df['Eccentricity'].abs().max()
ConvexArea = 32/original_df['ConvexArea'].abs().max()
EquivDiameter = 33/original_df['EquivDiameter'].abs().max()
Extent = 98/original_df['Extent'].abs().max()
Perimeter = 927/original_df['Perimeter'].abs().max()
Roundness = 677/original_df['Roundness'].abs().max()
AspectRation = 24/original_df['AspectRation'].abs().max()

my_prediction = model(torch.tensor([Area, MajorAxisLength, MinorAxisLength, Eccentricity,ConvexArea, EquivDiameter, Extent, Perimeter, Roundness,AspectRation], dtype = torch.float32).to(device))
```
And running this new sample
```python
if my_prediction <= 0.5:
    print("The new sample's class is Gonen")
else:
    print("The new sample's class is Jasmine")
```
```python
# The new sample's class is Gonen
```
