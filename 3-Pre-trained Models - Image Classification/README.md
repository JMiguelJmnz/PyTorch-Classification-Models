## Pre-Trained models for image classification

### Project Overview
Set up and use of pre-train models for image classification using a dataset of bean leaf lesions where the classification will fit in the categories healthy, angular_leaf_spot or bean_rust, implemented using a pytorch framework

### Importing libraries
```python
import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path
import opendatasets as od
```

### import dataset
The dataset consists of two splits: training and validation. It is downloaded directly from the Kaggle repository and loaded into pandas DataFrames:
```python
od.download("https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification")

base_path = Path(os.getcwd()).resolve()

train_df = pd.read_csv(base_path/"bean-leaf-lesions-classification"/"train.csv")
val_df = pd.read_csv(base_path/"bean-leaf-lesions-classification"/"val.csv")

train_df.head()
```
![image](https://github.com/user-attachments/assets/dcf0985c-09e3-431a-9474-fb9582f069ca)
To understand the distribution of classes in the training set, we can use:
```python
print(train_df["category"].value_counts())
```
which returns 
```python
2    348
1    345
0    341
```
As we can see, the dataset is relatively balanced across the three classes, which is beneficial for training classification models without introducing significant bias toward any particular category.

Now we set up our device for computation to speed up processing. If a GPU isn’t available, it falls back to the CPU:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available:", device)

if device.type == "cuda":
    print("GPU in use:", torch.cuda.get_device_name())
```
In my case is:
```python
Device available: cuda
GPU in use: NVIDIA GeForce RTX 3060
```
Before the training, a transformation pipeline prepares the images before feeding them into the model:
```python
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])
```
What is happening here is that the images are going to be resized to 128x128 pixel, then they are converted to PyTorch tensor, at the end its ensure the image tensor are explicitly converted into the correct type (float32)

Here we create a custom PyTorch dataset class that will allow how the data is loaded, transformed and served to a model
```python
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir  # e.g., base_path / "bean-leaf-lesions-classification"
        self.transform = transform
        self.labels = torch.tensor(dataframe["category"].values).to(device)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        relative_path = self.dataframe.iloc[idx, 0]  # or use column name if available
        full_path = self.root_dir / relative_path

        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image).to(device)

        label = self.labels[idx]
        return image, label
```
Now to create the datasets we instantiate them using the custom class defined before:
```python
img_dir = base_path / "bean-leaf-lesions-classification"

train_dataset = CustomImageDataset(dataframe=train_df, root_dir=img_dir, transform=transform)
val_dataset = CustomImageDataset(dataframe=val_df, root_dir=img_dir, transform=transform)
```
To preview a sample of the images from the training dataset we can run this loop:
```python
n_rows = 3
n_cols = 3

f, axarr = plt.subplots(n_rows, n_cols)

for row in range(n_rows):
    for col in range(n_cols):
        image_tensor = train_dataset[np.random.randint(0, len(train_dataset))][0]
        image = image_tensor.permute(1, 2, 0).cpu().numpy()  # H x W x C

        axarr[row, col].imshow(image)
        axarr[row, col].axis('off')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/d87dd02a-660c-4817-a229-19ba2fae50c2)

Here we define the key hyperparameters for our training loop, including the learning rate, batch size, and number of training epochs. These values can greatly influence the model’s performance and are often tuned based on validation results
```python
LR = 1e-3
BATCH_SIZE = 4
EPOCHS = 15
```
Here we use PyTorch’s DataLoader to batch and shuffle our training and validation datasets. This allows our model to process data in smaller, efficient chunks during training, and ensures each epoch sees the data in a different order — improving generalization.
```python
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
```
Next we load a pre-trained version of GoogLeNet with default weights  that comes built into PyTorch’s torchvision.models module.
```python
googlenet_model = models.googlenet(weights='DEFAULT')
```
By default the GoogLeNet model has 1000 out features, for our dataset we need to updated to our specific needs, so we can calculate the number of classes to predict and update the model before allocaiting it to the computation device.
```python
num_classes = len(train_df["category"].unique())
googlenet_model.fc = torch.nn.Linear(googlenet_model.fc.in_features, num_classes)
googlenet_model.to(device)
```
Set up the criterion used to calculate the loss using CrossEntropyLoss and set up the optimizer using Adam
```python
loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(googlenet_model.parameters(), lr = LR)
```
Now to train the model, the lists for plotting are initialized to be used at the end.
```python
total_loss_train_plot = []
total_acc_train_plot = []
```
Then we set at the start of each epoch the variables to accumulate the total values for each of the metrics and reset them to zero followed by the training and validation phases storing the results of the metrics at the e
```python
for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = googlenet_model(inputs)
        train_loss = loss_fun(outputs, labels)
        total_loss_train += train_loss.item()

        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/train_dataset.__len__()*100, 4))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/1000, 4)}, Train Accuracy: {round(total_acc_train/train_dataset.__len__()*100, 4)}")
```
![image](https://github.com/user-attachments/assets/6c7d4785-da15-40ef-879f-d45de68e5115)
At this point we can evaluate the accuracy of the model, setting torch.no_grad() to stop tracking gradients because there is no learning in this step
```python
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0

    for input, labels in val_loader:
        prediction = googlenet_model(input)

        acc = (torch.argmax(prediction, axis = 1) == labels).sum().item()
        total_acc_test += acc

print(round(total_acc_test/val_dataset.__len__()*100,2))
```
getting an accuracy of 92.48


### Transfer Learning
To test the difference, we can use the pretrained model only replacing the final layer to match our own classification and freeze the rest of the network so only the final layer gets trained.
```python
googlenet_transfer = models.googlenet(weights='DEFAULT')

for param in googlenet_transfer.parameters():
    param.requires_grad = False

googlenet_transfer.fc = torch.nn.Linear(googlenet_transfer.fc.in_features, num_classes)
googlenet_transfer.fc.requires_grad = True
googlenet_transfer.to(device)
```

```python
loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(googlenet_transfer.parameters(), lr = LR)

total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = googlenet_transfer(inputs)
        train_loss = loss_fun(outputs, labels)
        total_loss_train += train_loss.item()

        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/train_dataset.__len__()*100, 4))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/1000, 4)}, Train Accuracy: {round(total_acc_train/train_dataset.__len__()*100, 4)}")
```
![image](https://github.com/user-attachments/assets/a8daffd4-ecdf-40a1-823a-f8fedff68787)

```python
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0

    for input, labels in val_loader:
        prediction = googlenet_transfer(input)

        acc = (torch.argmax(prediction, axis = 1) == labels).sum().item()
        total_acc_test += acc

print(round(total_acc_test/val_dataset.__len__()*100,2))
```
getting an accuracy of 75.19

Whit this we can see the improvement of the fine tuned model.  
To test a new image we could use:
```python
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).to(device)

    googlenet_model.eval()

    with torch.no_grad():
        output = googlenet_model(image.unsqueeze(0))

    output = torch.argmax(output, axis=1).item()
    return label_encoder.inverse_transform([output])
```
And to run a new sample
![bean_rust_sample](https://github.com/user-attachments/assets/660ff2b2-9b2c-4343-9ae7-c0e928eeb59c)
```python
predict_image("bean_rust_sample.jpg").item()
```
Getting a prediction of category 2 - bean rust
