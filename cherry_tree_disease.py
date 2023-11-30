import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data.dataset import random_split
from torchvision.transforms import ToTensor, Grayscale, Compose, Normalize, CenterCrop, Resize

#Load the data into a Tensor divided into the 5 different classes
data_transforms = Compose([Grayscale(), ToTensor(), Resize((64,64),antialias=True)])
all_data = datasets.ImageFolder("13-07-2022/13_07_2022/Ground_RGB_Photos",transform=data_transforms)
print(all_data.class_to_idx)
#Split the data into Training and Validation sets
split_data = random_split(all_data,[0.9,0.1])
train_data = split_data[0]
valid_data = split_data[1]


print(len(train_data))
print(len(valid_data))
image, label = train_data.__getitem__(0)
print(image.shape)
print(label)

#Find each unique class in the training data
labels = []
for image, label in train_data:
  labels.append(label)
labels_unique, counts = np.unique(labels, return_counts=True)

#Sum up how many items of each class there are in the training data
class_weights = np.array([sum(counts)/c for c in counts])
train_weights = [class_weights[e] for e in labels]

#Find each unique class in the validation data
labels = []
for image, label in valid_data:
  labels.append(label)
labels_unique, counts = np.unique(labels, return_counts=True)

#Sum up how many items of each class there are in the validation data
class_weights = np.array([sum(counts)/c for c in counts])
valid_weights = [class_weights[e] for e in labels]

#Create Weighted samplers for both sets of data that have the same size as the full set of data
train_sampler = torch.utils.data.WeightedRandomSampler(train_weights,len(train_data))
valid_sampler = torch.utils.data.WeightedRandomSampler(valid_weights,len(valid_data))

#Create a Data Loader for the training data
train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=152, num_workers=1)

class network(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=5,stride=1,padding=2)
    self.conv2 = nn.Conv2d(in_channels=4,out_channels=8,kernel_size=5,stride=1,padding=2)
    self.conv3 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5,stride=1,padding=2)
    self.conv4 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5,stride=1,padding=2)
    self.conv5 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
    self.conv6 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)
    self.conv7 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
    self.conv8 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)
    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(16)
    self.bn3 = nn.BatchNorm2d(32)
    self.bn4 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(in_features=4*4*64,out_features=256)
    self.fc2 = nn.Linear(in_features=256,out_features=64)
    self.fc3 = nn.Linear(in_features=64,out_features=16)
    self.fc4 = nn.Linear(in_features=16,out_features=3)
    self.bn5 = nn.BatchNorm1d(256)
    self.bn6 = nn.BatchNorm1d(64)
    self.bn7 = nn.BatchNorm1d(16)
    self.relu = nn.ReLU()

  def forward(self, x):

    #B x 1 x 64 x 64
    x = self.conv1(x)
    x = self.relu(x)
    #B x 4 x 64 x 64
    x = self.conv2(x)
    x = self.bn1(x)
    x = self.relu(x)
    #B x 8 x 64 x 64
    x = self.maxpool(x)
    
    #B x 8 x 32 x 32
    x = self.conv3(x)
    x = self.bn2(x)
    x = self.relu(x)
    #B x 16 x 32 x 32
    x = self.conv4(x)
    x = self.bn2(x)
    x = self.relu(x)
    #B x 16 x 32 x 32
    x = self.maxpool(x)
    
    #B x 16 x 16 x 16
    x = self.conv5(x)
    x = self.bn3(x)
    x = self.relu(x)
    #B x 32 x 16 x 16
    x = self.conv6(x)
    x = self.relu(x)
    #B x 32 x 16 x 16
    x = self.maxpool(x)
    
    #B x 32 x 8 x 8
    x = self.conv7(x)
    x = self.bn4(x)
    x = self.relu(x)
    #B x 64 x 8 x 8
    x = self.conv8(x)
    x = self.relu(x)
    #B x 64 x 8 x 8
    x = self.maxpool(x)

    #B x 64 x 4 x 4
    x = x.view(x.size(0), -1)

    #Flattened to B x 1024
    x = self.fc1(x)
    x = self.relu(x)
    #B x 512
    x = self.bn5(x)
    x = self.fc2(x)
    x = self.relu(x)
    #B x 128
    x = self.bn6(x)
    x = self.fc3(x)
    #B x 32
    x = self.bn7(x)
    x = self.fc4(x)

    #B x 3
    return x

#Initialise the network
model = network()


#Enable the use of the GPU for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
learningRate = 0.01
#Set up the SGD optimiser
optim = torch.optim.SGD(model.parameters(),lr=learningRate)
iterations_per_epoch = len(train_loader)

# Set up the loss function for multiclass classification
loss_func = nn.CrossEntropyLoss()

#Move the model and loss function onto the GPU
model.to(device)
loss_func.to(device)

for epoch in range(num_epochs):
  total_loss = 0
  for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    optim.zero_grad()
    loss = loss_func(output,labels)
    loss.backward()
    optim.step()
    if (i+1) % 25 == 0:
       print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, iterations_per_epoch, loss.item()))
    total_loss += loss
  print('Total loss over epoch {}: {:.2f}'.format(epoch+1,total_loss))



correct = 0
total = 0
#Set up the data loader for the validation set
valid_loader = torch.utils.data.DataLoader(valid_data, sampler=valid_sampler, batch_size=152, num_workers=1)

#Iterate through the validation set and record which images the model is able to correctly classify
for images, labels in valid_loader:
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    pred_y = torch.argmax(output, 1)
    correct += (pred_y == labels).sum()
    total += float(labels.size(-1))
    
print(correct.item(),"/",int(total))

accuracy = correct/total
print('Validation Accuracy of the model: %.2f' % accuracy)

torch.save(model.state_dict(), 'weights.pkl')