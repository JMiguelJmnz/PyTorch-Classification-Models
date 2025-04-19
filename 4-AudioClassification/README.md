## Audio Classification using pytorch

### Project Overview
This project implements an end-to-end audio classification pipeline using a dataset of Quran recitations.

### Importing libraries
```python
import torch
from torch import nn
from torch.optim import Adam
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from skimage.transform import resize
from pathlib import Path
```

### import dataset
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification")
```
The dataset is sourced from Kaggle: Quran Recitations for Audio Classification. It includes .wav files labeled by reciter name.

Now we set up our device for computation to speed up processing. If a GPU isn’t available, it falls back to the CPU:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Is Cuda available?: ", torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name())  # Should print your GPU name
```

Creation of a dataframe with relative paths to audio files
```python
base_path = Path(os.getcwd()).resolve()
data_df = pd.read_csv(base_path/"quran-recitations-for-audio-classification"/"files_paths.csv")
```
Then update the FilePath column by prepending the full directory path to each relative file name. This is necessary for accessing the audio files correctly in later processing steps.
```python
data_df["FilePath"] = data_df["FilePath"].apply(lambda x: base_path / "quran-recitations-for-audio-classification" / "Dataset"/ x)
```
We can do a quick plot to see the data distribution
```python
print("Data Shape: ", data_df.shape)
print("Class Distribution: ", data_df["Class"].value_counts())

print()

plt.figure(figsize= (8,8))

plt.pie(data_df["Class"].value_counts(), labels = data_df["Class"].value_counts().index, autopct='%1.1f%%')

```
![image](https://github.com/user-attachments/assets/a7975177-c909-4f5c-b9e0-3a659d0a735c)

Now we use LabelEncoder() from scikit-learn to convert the string values into numerical code and then the training, validation and testing groups are defined
```python
label_encoder = LabelEncoder()
data_df["Class"] = label_encoder.fit_transform(data_df["Class"])

train = data_df.sample(frac=0.7, random_state= 7)
test = data_df.drop(train.index)

val = test.sample(frac=0.5, random_state= 7)
test = test.drop(val.index)
```

This custom Dataset class is used to load and preprocess audio files as mel spectrograms, which are then returned with their corresponding labels for model training.
```python
class CustomAudioDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe        
        self.audios = [torch.Tensor(self.get_spectogram(path)).type(torch.FloatTensor) for path in dataframe["FilePath"]]

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = int(self.dataframe.iloc[idx]["Class"])
        label = torch.tensor(label, dtype=torch.long).to(device)
        audio = self.audios[idx].unsqueeze(0).to(device)
        return audio, label
    
    def get_spectogram(self, file_path):
        sr = 22050
        duration = 5

        img_height = 128
        img_width = 256

        signal, sr = librosa.load(str(file_path), sr=sr, duration = duration)
        spec = librosa.feature.melspectrogram(y = signal, sr = sr, n_fft=2048, hop_length = 512, n_mels = 128)
        spec_db = librosa.power_to_db(spec, ref = np.max)

        spec_resized = librosa.util.fix_length(spec_db, size = (duration*sr) // 512+1)
        spec_resized = resize(spec_resized, (img_height, img_width), anti_aliasing=True)
        return spec_resized
```
Using the Test, validation and test groups, we create the datasets with the CustomAudioDataset() class created
```python
train_dataset = CustomAudioDataset(dataframe = train)
val_dataset = CustomAudioDataset(dataframe = val)
test_dataset = CustomAudioDataset(dataframe = test)
```
Definition of the hyperparameters
```python
# Hyperparameters
LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 25
```
And now we set up the DataLoaders to set the batch size of each group and to enable shuffle in the train data to randomly mix the data at each epoch so the model doesn’t get used to the order of samples
```python
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)
```
The Convolutional Neural Network (CNN) for audio classification will use mel spectrograms, where each image-like representation of an audio file passes through the network for classification.
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.pooling = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear((64*16*32), 4096)
        self.linear2 = nn.Linear(4096, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, len(data_df["Class"].unique()))

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = self.dropout(x)
        
        x = self.linear3(x)
        x = self.dropout(x)

        x = self.output(x)
        return x
```
Then we allocate the model to the computing device, in this case, CUDA
```python
model = Net().to(device)
```
The criterion and optimizer are defined before the training
```python
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = LR)
```
And then we train the model while storing the loss and accuracy to visualize the model eficency later
```python
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

import time
start_time = time.time()

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    for input, labels in train_loader:
        outputs = model(input) 
        train_loss = criterion(outputs, labels)
        total_loss_train += train_loss.item()
        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_train += train_acc

        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        for input, labels in val_loader:
            outputs = model(input)
            val_loss = criterion(outputs, labels)
            total_loss_val += val_loss.item()

            val_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
            total_acc_val += val_acc

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_validation_plot.append(round(total_loss_val/1000, 4))

    total_acc_train_plot.append(round((total_acc_train/train_dataset.__len__()) * 100, 4))
    total_acc_validation_plot.append(round((total_acc_val/val_dataset.__len__()) * 100, 4))

    print(f"Epoch: {epoch + 1}/{EPOCHS}, Train Loss: {round(total_loss_train/100, 4)}, Train Accuracy: {round((total_acc_train/train_dataset.__len__()) * 100, 4)}, Validation Loss: {round(total_loss_val/100, 4)}, Validation Accuracy: {round((total_acc_val/val_dataset.__len__()) * 100, 4)}")
    print("="*30)

print("Training Time is: ", round(time.time() - start_time), 4, "Seconds")
```
In my case, the training took 345 seconds and by the fluctuation of the loss and accuracy, we can think that it might be overfitting
![image](https://github.com/user-attachments/assets/645514d2-53b2-4f4f-8bae-14607a470285)

Now we measure the total Accuracy
```python
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    for input, labels in test_loader:
        prediction = model(input)
        acc = (torch.argmax(prediction, axis = 1) == labels).sum().item()

        total_acc_test += acc

print(f"Total Accuracy Score is: ", {round((total_acc_test/test_dataset.__len__() * 100), 4)})
```
Getting 95%, but to confirm if our results are correct, we can visualize the results
```python
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(15,5))

axs[0].plot(total_loss_train_plot, label = "Training Loss")
axs[0].plot(total_loss_validation_plot, label = "Validation Loss")
axs[0].set_title("Training and Validation Loss over Epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
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
![image](https://github.com/user-attachments/assets/23be1d31-1860-4015-956a-68bc9fbae744)

