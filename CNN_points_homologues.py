#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 19:15:09 2018

@author: guillemettefonteix
"""

##############################################################################
#Import des librairies
##############################################################################

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

from matplotlib.pylab import arange
import matplotlib.pyplot as plt

import numpy as np
import PIL
import os
import itertools
from PIL import Image

from sklearn.metrics import confusion_matrix, roc_curve

  
###############################################################################
### Parameters and paths
###############################################################################
    
num_epochs = 15
num_classes = 2
batch_size = 40
learning_rate = 0.0005
    
classes = ('homologue', 'non-homologue')
    
path = os.path.abspath('/Users/guillemettefonteix/Desktop/projet_informatique/code/Etape_2/data/eTPR_GrayMax/eTVIR_ACGT') 
path_model = os.path.abspath('/Users/guillemettefonteix/Desktop/projet_informatique/code/Etape_2/final/pytorch/cnn/model.ckpt') 

###############################################################################
## Mean and std for normalization
###############################################################################

def mean_std():
    """
    Calcul la moyenne et l'écart type du jeu de données pour la normalisation
    """
    traindata = torchvision.datasets.ImageFolder(path, transforms.ToTensor())
    
    image_means = torch.stack([t.mean(1).mean(1) for t, c in traindata])   
    mean = np.asarray(image_means.mean(0)[0])
    
    image_std = torch.stack([t.std(1).std(1) for t, c in traindata])
    std = np.asarray(image_std.std(0)[0])
    
    return(mean,std)
 
#mean, std = mean_std()
#print(mean, std)

###############################################################################
## Load and normalizing the images training and test datasets using torchvision
###############################################################################

mean = 0.52598995
std  = 0.019920329

#The compose function allows for multiple transforms
transform = transforms.Compose([
    PIL.ImageOps.grayscale,                     #RGB to Gray scale
    transforms.ToTensor(),                      #transforms.ToTensor() converts our PILImage to a tensor of shape 
    transforms.Normalize([mean], [std]),        #transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) 
    ])    
    
dataset = torchvision.datasets.ImageFolder(path, transform=transform)


# 60% of the data for the training set, 20% for the test and 20% for the validation
train_length = int(0.6* len(dataset))
valid_length  = int(0.2* len(dataset))
test_length = len(dataset)- train_length - valid_length

train_loader, valid_loader, test_loader = torch.utils.data.random_split(dataset, (train_length,valid_length, test_length))

print("There are {} images in the training set".format(len(train_loader)))
print("There are {} images in the test set".format(len(test_loader)))
print("There are {} images in the validation set".format(len(valid_loader)))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                      batch_size=batch_size, 
                                      shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_loader,
                                      shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_loader,
                                      shuffle=True)



print("Loading completed successfully!")


##########################################################################
# Define a Convolutional Neural Network (CNN)
##########################################################################
     
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Our batch shape for input x is (1, 10, 20)
        #Convolution; Input channels = 1 (grayscale), output channels = 27
        self.conv1 = nn.Conv2d(1, 27, kernel_size=3, stride =1, padding =1)
        self.conv2 = nn.Conv2d(27, 27, kernel_size=3, stride =1, padding =1)
        
        #Fully connected
        self.fc1 = nn.Linear(27 * 5 * 10, 400)
        self.fc2 = nn.Linear(400, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)
     
        self.drop_out = nn.Dropout()   
        
    def forward(self, x): 
        #Size changes from (1, 10, 20) to (27, 10, 20)
        x = F.relu(self.conv1(x))
        #Size changes from  (27, 10, 20) to (27, 5, 10)
        x = F.max_pool2d(x, (2, 2))
        #Size changes from (27, 5, 10) to (27, 5, 10)
        x = F.relu(self.conv2(x))
        #Size changes from (27, 5, 10) to (1, 27*5*10)
        
        x = x.view(-1, self.num_flat_features(x))     #self.num_flat_features(x) = 27*5*10

        #Computes the activation of the first fully connected layer
        #Size changes from (1, 27*5*10) to (1, 400)
        x = F.relu(self.fc1(x))    
        #Computes the second fully connected layer 
        #Size changes from (1, 400) to (1, 84)
        x = F.relu(self.fc2(x))   
        #Computes the second fully connected layer 
        #Size changes from (1, 84) to (1, 10)
        x = F.relu(self.fc3(x))  
        #allow to avoid overfitting
        x = self.drop_out(x)
        
        #Computes the fourth fully connected layer (activation applied later)
        #Size changes from (1, 10) to (1, 2)
        x = self.fc4(x)
        
        return x
    
    

    def num_flat_features(self, x):
        size = x.size()[1:]       # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net()

###########################################################################
## Loss and optimizer
###########################################################################

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##############################################################################
## Train the network on the training data
##############################################################################

total_step = len(train_loader)

loss_list = []
valid_list =[]

with torch.no_grad():
        correct = 0
        total = 0
        labels_valid = []
        labels_predicted = []
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            b = np.asarray(labels)
            for i in b:
                labels_valid.append(i)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 
             
            a = np.asarray(predicted)
            
            for i in a:
                labels_predicted.append(i)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
          
print('Test Accuracy of the model on the test set: {} %'.format(100 * correct / total))
valid_list.append(100 * correct / total)
    
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
   
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)   # récupère la position du label
            
        a = np.asarray(predicted)

        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    with torch.no_grad():
        correct = 0
        total = 0
        labels_test = []
        labels_predicted = []
        
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            b = np.asarray(labels)
            for i in b:
                labels_test.append(i)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            a = np.asarray(predicted)
            for i in a:
                labels_predicted.append(i)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
          
    print('Valid Accuracy of the model on the validation set: {} %'.format(100 * correct / total))
    valid_list.append(100 * correct / total)


#On affiche les variations de la fonction de perte; le but est de minimiser cette valeur
#La fonction doit donc être décroissante
y = loss_list

plt.plot(y)  
plt.ylabel('Loss')
plt.show() 
     
##############################################################################
### Test the network on the test data
##############################################################################

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    labels_test = [] 
    labels_predicted = []    # liste de 0 et de 1 (homologue et non-homologue) 
    
    y_score = np.zeros((1,2))
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        b = np.asarray(labels)
       
        for i in b:
            labels_test.append(i)
    
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) #résultat: indique homologue ou non-homologue
        
        
        s = np.asarray(outputs)
        
        for i in s:
            #On enregistre les scores obtenus pour chaque image pour le calcul de la courbe ROC
            u = np.zeros((1,2))
            u[0][0] = i[0]
            u[0][1] = i[1]
            y_score = np.concatenate((y_score, u), axis=0)
  
        a = np.asarray(predicted)
       
        for i in a:
            labels_predicted.append(i)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    y_score  = y_score[1:]
    print('Test Accuracy of the model on the  test set: {} %'.format(100 * correct / total))
    
    
##############################################################################
#####  Confusion matrix  
##############################################################################
        
cnf_matrix = confusion_matrix(labels_test, labels_predicted)
cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, cmap=plt.cm.Blues)
plt.grid('off')
plt.colorbar()
plt.title('CNN')
tick_marks = np.arange(2)
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
    
fmt = '.4f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
plt.tight_layout()
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig('confusion.png')
plt.show()

###############################################################################
###### ROC curve 
###############################################################################

fpr_cl = dict()
tpr_cl = dict()

y_pred = np.asarray(labels_predicted)
y_proba = y_score
y_test = np.asarray(labels_test)


fpr_cl["classe 0"], tpr_cl["classe 0"], _ = roc_curve(
    y_test == 0, y_proba[:, 0].ravel())
fpr_cl["classe 1"], tpr_cl["classe 1"], _ = roc_curve(
    y_test, y_proba[:, 1].ravel())  # y_test == 1


prob_pred = np.array([y_proba[i, 1 if c else 0]
                         for i, c in enumerate(y_pred)])
fpr_cl["tout"], tpr_cl["tout"], _ = roc_curve(
    (y_pred == y_test).ravel(), prob_pred)


plt.figure()
for key in fpr_cl:
    plt.plot(fpr_cl[key], tpr_cl[key], label=key)

lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Proportion mal classée")
plt.ylabel("Proportion bien classée")
plt.title('ROC(s) avec predict_proba')
plt.legend(loc="lower right")
plt.savefig('roc.png')
#
################################################################################
####### Save the model 
################################################################################

torch.save(model, path_model)

##############################################################################
##### Réutilisation du modèle enregistré
##############################################################################

def image_loader(path_im):
    """
    path_im: chemin vers l'image à classifier 
    return: image ayant subit les mêmes transformations que lors de l'apprentissage
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    image = Image.open(path_im)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device) #assumes that you're using GPU

def reutilisation(path_im):
    """
    path_im : chemin vers l'image à classifier 
    return: homologue ou non-homologue 
    """
    model = torch.load(path_model)
    model.eval()
    image = image_loader(path_im)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return("Ces points sont {}s ".format(classes[predicted]))

 
#print(reutilisation('/Users/guillemettefonteix/Desktop/projet_informatique/code/eval/12n.png'))