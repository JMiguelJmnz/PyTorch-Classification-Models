### Text Classification Using PyTorch

### Project Overview
This project involves building a deep learning model to perform image classification using PyTorch framework.
<br>The goal is to develop a neural network capable of automatically classifying images of animals into three predefined categories: dog, cat, and wild, based on their content.

### Importing libraries
For this project, theses are the libraries used
```python
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
```
As usual, the compute device is set to GPU to speed up training
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Is Cuda available?: ", torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name())  # Should print your GPU name
```
The dataset is downloaded using opendatasets
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection")
```
And its loaded and cleaned a little bit to start using it
```python
base_path = Path(os.getcwd()).resolve()
data_df = pd.read_json(base_path/"news-headlines-dataset-for-sarcasm-detection"/"Sarcasm_Headlines_Dataset.json",lines=True)
data_df.dropna(inplace = True)
data_df.drop_duplicates(inplace=True)
data_df.drop(["article_link"], inplace = True, axis = 1)
print(data_df.shape)
```
It returns 26708 rows with 2 columns, which are the headline and if is sarcastic represented with 0 as not sarcastic and 1 as is sarcastic

Using the train_test_split module from sklearn, the data is splitted into the train, validation and testing
```python
X_train, X_test, y_train, y_test = train_test_split(np.array(data_df["headline"]), np.array(data_df["is_sarcastic"]), test_size = 0.3)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

print("Training size: ", X_train.shape[0], "rows, which is: ", round(X_train.shape[0]/data_df.shape[0], 4) *100, "%")
print("Validation size: ", X_val.shape[0], "rows, which is: ", round(X_val.shape[0]/data_df.shape[0], 4) *100, "%")
print("Test size: ", X_test.shape[0], "rows, which is: ", round(X_test.shape[0]/data_df.shape[0], 4) *100, "%")
```
Splitting the groups as follos:  
Training size:  18695 rows, which is:  70.0 %  
Validation size:  4006 rows, which is:  15.0 %  
Test size:  4007 rows, which is:  15.0 %


To process raw text for input into a model, it must first be converted into a numerical format called tokens. This is done using AutoTokenizer from the transformers library by Hugging Face. The corresponding AutoModel is then used to load the pretrained BERT model, allowing us to leverage its language understanding capabilities.

Both components can be loaded from Hugging Face’s model hub using:
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
```

A custom PyTorch Dataset class is defined to preprocess the text and prepare it for training. The class tokenizes each input string using the pretrained tokenizer, pads or truncates to a fixed length, and moves both inputs and labels to the appropriate device.
```python
class dataset(Dataset):
    def __init__(self, X,Y):
        self.X = [tokenizer(x,
                            max_length = 100,
                            truncation = True,
                            padding = "max_length",
                            return_tensors = "pt").to(device)
                            for x in X
        ]
        self.Y = torch.tensor(Y, dtype = torch.float32).to(device)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, indx):
        return self.X[indx], self.Y[indx]
```
Usage:
```python
training_data = dataset(X_train, y_train)
validation_data = dataset(X_val, y_val)
testing_data = dataset(X_test, y_test)
```

At this point, the hyperparameters can be defined for the training step
```python
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
```
The custom datasets are wrapped in PyTorch DataLoaders to enable mini-batch training. Shuffling is enabled only for the training set to randomize the order of samples, helping the model generalize better instead of memorizing sequence patterns.
```python
train_dataloader = DataLoader(training_data, batch_size = BATCH_SIZE, shuffle = True)
validation_dataloader = DataLoader(validation_data, batch_size = BATCH_SIZE, shuffle = False)
testing_dataloader = DataLoader(testing_data, batch_size = BATCH_SIZE, shuffle = False)
```
The neural network is defined by creating a custom class that inherits from PyTorch’s nn.Module. Inside the class, we initialize the model layers in the constructor and define how data flows through them in the forward() method. This includes passing input through the pretrained BERT model, followed by dropout and linear layers for classification.
```python
class MyModel(nn.Module):
    def __init__(self, bert):
        super(MyModel, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear (768, 384)
        self.linear2 = nn.Linear(384,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask, return_dict = False)[0][:,0]
        output = self.linear1(pooled_output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output
```
Before training the model, requires_grad is set to False to allow the model to focus on training only the custom layers added on top of BERT to use BERT mostly as a feature extractor and then make sure is running on the same device as the rest of the data (GPU)
```python
for param in bert_model.parameters():
    param.requires_grad = False

model = MyModel(bert_model).to(device)
```

Now we set up the loss function and optimizer for training.
BCELoss (Binary Cross Entropy) is used because the task is binary classification, producing a probability between 0 and 1.
We use the Adam optimizer, a commonly used variant of stochastic gradient descent that adapts learning rates per parameter.
The learning rate (LR) controls the step size for updates during training.
```python
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr = LR)
```
Now we train the neural network over several epochs and track both loss and accuracy on the training and validation sets. This helps evaluate how well the model is learning and whether it's generalizing or overfitting.
```python
total_loss_train_plot = []
total_acc_validation_plot = []
total_acc_train_plot = []
total_loss_validation_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0
    for indx, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        prediction = model(inputs["input_ids"].squeeze(1), inputs["attention_mask"].squeeze(1)).squeeze(1)
        batch_loss = criterion(prediction, labels)
        total_loss_train += batch_loss.item()

        acc = (prediction.round() == labels).sum().item()

        total_acc_train += acc

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        for indx, data in enumerate(validation_dataloader):
            inputs, labels = data
            inputs.to(device)
            labels.to(device)

            prediction = model(inputs["input_ids"].squeeze(1), inputs["attention_mask"].squeeze(1)).squeeze(1)
            batch_loss = criterion(prediction, labels)
            total_loss_val += batch_loss.item()

            acc = (prediction.round() == labels).sum().item()
            total_acc_val += acc

    total_loss_train_plot.append(round(total_loss_train / len(train_dataloader), 4))
    total_loss_validation_plot.append(round(total_loss_val/1000, 4))
    total_acc_train_plot.append(round((total_acc_train/training_data.__len__()) * 100, 4))
    total_acc_validation_plot.append(round((total_acc_val/validation_data.__len__()) * 100, 4))

    print(f"""
    Epoch No. {epoch + 1}, Train Loss: {round(total_loss_train/1000, 4)}, Train Accuracy: {round((total_acc_train/training_data.__len__()) * 100, 4)},
          Validation loss: {round(total_loss_val/1000, 4)}, Validation Accuracy: {round((total_acc_val/validation_data.__len__()) * 100, 4)}
          """)
```
Finally, we evaluate how well the trained model performs on the unseen test data by computing the accuracy and loss, using torch.no_grad() to disable gradient calculations and speed up inference.
```python
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    
    for indx, data in enumerate(testing_dataloader):
        inputs, labels = data
        inputs.to(device)
        labels.to(device)

        prediction = model(inputs["input_ids"].squeeze(1), inputs["attention_mask"].squeeze(1)).squeeze(1)
        batch_loss = criterion(prediction, labels)
        total_loss_test += batch_loss.item()

        acc = (prediction.round() == labels).sum().item()

        total_acc_test += acc

print(f"Accuracy Score on testing Data is: {round((total_acc_test/testing_data.__len__()) * 100, 4)}" )
```
Getting an accuracy of 86% is a strong starting point, but there’s room for improvement. To better understand the model’s learning behavior and potential overfitting or underfitting, we can visualize how the loss and accuracy evolved across epochs for both the training and validation sets. These plots give insights into the model's generalization and help guide future tuning or architecture changes
```python
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(15,5))

axs[0].plot(total_loss_train_plot, label = "Training Loss")
axs[0].plot(total_loss_validation_plot, label = "Validation Loss")
axs[0].set_title("Training and Validation Loss over Epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].set_ylim([0,1])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label = "Training Accuracy")
axs[1].plot(total_acc_validation_plot, label = "Validation Accuracy")
axs[1].set_title("Training and Validation Accuracy over Epochs")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].set_ylim([0,100])
axs[1].legend()

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/85f392ae-c751-431e-ac13-f7c7a8a6474a)
