#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      models.py
# Create:        01/22/22
# Description:   Train and test dataset on models
#---------------------------------------------------------------------

# IMPORTS:
# Python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import itertools
import datetime
import time
import seaborn as sn

# sklearn libraries
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# pytorch libraries
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# FUNCTIONS:

class Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, transform = None):
        'Initialization'
        self.df = df
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y



def train(train_loader, model, loss_function, optimizer, device):
    model.train()

    size = len(train_loader.dataset)
    num_batches = len(train_loader)

    train_loss = 0
    train_accuracy = 0

    for batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Compute prediction error
        pred = model(images)
        loss = loss_function(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    train_accuracy /= size
    print(f"Train Error: [Accuracy: {(100 * train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f}]\n")

    return train_loss, train_accuracy



def test(test_loader, model, loss_function, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    model.eval()

    num_batches = len(test_loader)
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Compute Prediction Error
            pred = model(images)
            loss = loss_function(pred, labels)
            test_loss += loss.item()
            test_accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    test_accuracy /= size
    print(f"Test Error: [Accuracy: {(100 * test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f}]")

    return test_loss, test_accuracy



def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#---------------------------------------------------------------------
# Function:    model()
# Description: Train and test dataset on models
#---------------------------------------------------------------------
def model(file, skin_df_train, skin_df_test, number_Cell_Type):
    print(">>> START MODEL")

    learning_rate = 1e-3
    batch = 32
    num_worker = 4
    epoch_num = 2 #10
    input_size = 225
    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)
    num_classes = number_Cell_Type
    feature_extract = False
    use_pretrained = True

    # Define the device (use GPU if avaliable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)

    # TODO: Create seperate function
    model = models.resnet50(pretrained=True) # Load Resnet50
    # print(model.fc)
    num_ftrs = model.fc.in_features # Find number of features in last layer
    model.fc = torch.nn.Linear(num_ftrs, 7) # Adjust last layer to 7 different classes
    # print(model.fc)

    model = model.to(device) # Add device to model

    # Transformations
    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
                                        transforms.ColorJitter(brightness = 0.1, contrast = 0.1, hue = 0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

    training_set = Dataset(skin_df_train, transform = train_transform)
    train_loader = DataLoader(training_set, batch_size = batch, shuffle = True, num_workers = num_worker)

    test_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    test_set = Dataset(skin_df_test, transform = test_transform)
    test_loader = DataLoader(test_set, batch_size = batch, shuffle = False, num_workers = num_worker)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # Main Training and Testing Function
    best_test_accuracy = 0
    total_train_loss = []
    total_train_accuracy = []
    total_test_loss = []
    total_test_accuracy = []
    train_time_list = []
    test_time_list = []
    total_time_list = []

    for epoch in range(epoch_num):
        start_time = time.time()

        print('EPOCH {}:'.format(epoch))

        train_loss, train_accuracy = train(train_loader, model, loss_function, optimizer, device)
        total_train_loss.append(train_loss) # Average Training Loss
        total_train_accuracy.append(train_accuracy) # Average Training Accuracy

        mid_time = time.time()
        training_time_difference = mid_time - start_time
        train_time_list.append(training_time_difference)

        test_val, test_accuracy = test(test_loader, model, loss_function, device)
        total_test_loss.append(test_val) # Average Test Loss
        total_test_accuracy.append(test_accuracy) # Average Test Accuracy

        end_time = time.time()
        test_time_difference = end_time - mid_time
        test_time_list.append(test_time_difference)
        total_time_difference = end_time - start_time
        total_time_list.append(total_time_difference)


    print("Total Training Loss: ", total_train_loss)
    print("Total Training Loss: ", np.mean(total_train_loss))

    print("Total Training Accuracy: ", total_train_accuracy)
    print("Total Training Accuracy: ", total_train_accuracy)

    print("Total Testing Loss: ", total_train_loss)
    print("Total Testing Loss: ", total_train_loss)

    print("Total Testing Accuracy: ", total_test_accuracy)
    print("Total Testing Accuracy: ", total_test_accuracy)

    print("Training Time List: ", train_time_list)
    print("Training Time List: ", train_time_list)

    print("Test Time List: ", test_time_list)
    print("Test Time List: ", test_time_list)

    print("Total Time List: ", total_time_list)
    print("Total Time List: ", total_time_list)

    # Model Evaluation
    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.legend()
    fig.savefig('OUTPUT\Figures\Training_Loss_vs_Training_Accuracy.png', bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_test_loss, label = 'Testing Loss')
    plt.plot(total_test_accuracy, label = 'Testing Accuracy')
    plt.legend()
    fig.savefig('OUTPUT\Figures\Testing_Loss_vs_Testing_Accuracy.png', bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.plot(total_test_loss, label = 'Testing Loss')
    plt.legend()
    fig.savefig('OUTPUT\Figures\Training_Loss_vs_Testing_Loss.png', bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.plot(total_test_accuracy, label = 'Testing Accuracy')
    plt.legend()
    fig.savefig('OUTPUT\Figures\Training_Accuracy_vs_Testing_Accuracy.png', bbox_inches = 'tight')



    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype = torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype = torch.long, device='cpu')

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, classes.view(-1).cpu()])

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

    # plot the confusion matrix
    plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
    plot_confusion_matrix(conf_mat, plot_labels)
    plt.show()

    # Generate Classification Report
    report = classification_report(lbllist.numpy(), predlist.numpy(), target_names = plot_labels)
    print(report)




    print(">>> END Model")
