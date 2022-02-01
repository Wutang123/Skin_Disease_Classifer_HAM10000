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



def train(file, train_loader, model, loss_function, optimizer, device):
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
    print(f"Training Error: [Accuracy: {(100 * train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f}]")
    file.write("Training Error: [Accuracy: {:>0.5f}%, Avg Loss: {:>8f}]\n".format((100 * train_accuracy), train_loss))

    return train_loss, train_accuracy



def test(file, test_loader, model, loss_function, device):
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
    print(f"Testing Error:  [Accuracy: {(100 * test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f}]\n")
    file.write("Testing Error:  [Accuracy: {:>0.5f}%, Avg Loss: {:>8f}]\n\n".format((100 * test_accuracy), test_loss))

    return test_loss, test_accuracy



def plot_confusion_matrix(conf_mat, plot_labels, save_fig):
    normalize = False

    fig = plt.figure()
    plt.imshow(conf_mat, interpolation='nearest', cmap = plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(plot_labels))
    plt.xticks(tick_marks, plot_labels, rotation = 45)
    plt.yticks(tick_marks, plot_labels)

    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, conf_mat[i, j], horizontalalignment = "center", color="white" if conf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if (save_fig):
        fig.savefig('OUTPUT\Figures\Confusion_Matrix.png', bbox_inches = 'tight')



#---------------------------------------------------------------------
# Function:    model()
# Description: Train and test dataset on models
#---------------------------------------------------------------------
def model(file, save_fig, skin_df_train, skin_df_test, number_Cell_Type):

    learning_rate = 1e-3
    batch = 32
    num_worker = 4
    epoch_num = 1 #10
    input_size = 225
    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)
    num_classes = number_Cell_Type
    feature_extract = False
    use_pretrained = True

    print("Learing Rate:       {}".format(learning_rate))
    file.write("Learing Rate:       {} \n".format(learning_rate))

    print("Batch Size:         {}".format(batch))
    file.write("Batch Size:         {} \n".format(batch))

    print("Number of Workers:  {}".format(num_worker))
    file.write("Number of Workers:  {} \n".format(num_worker))

    print("Epoch Number:       {}".format(epoch_num))
    file.write("Epoch Number:       {} \n".format(epoch_num))

    print("Image Size:         {} by {}".format(input_size, input_size))
    file.write("Image Size:         {} by {} \n".format(input_size, input_size))

    print("Normalized Mean:   ", norm_mean)
    file.write("Normalized Mean:    " + str(norm_mean) + "\n")

    print("Normalized STD:    ", norm_std)
    file.write("Normalized STD:     " + str(norm_std) + "\n")

    # Define the device (use GPU if avaliable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Used:       ", device, "\n")
    file.write("Device Used:        " + str(device) + "\n\n")

    # TODO: Create seperate function
    model = models.resnet50(pretrained = use_pretrained) # Load Resnet50
    num_ftrs = model.fc.in_features # Find number of features in last layer
    model.fc = torch.nn.Linear(num_ftrs, num_classes) # Adjust last layer to 7 different classes

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

        print("EPOCH {}:".format(epoch))
        file.write("EPOCH: " + str(epoch) + "\n")

        train_loss, train_accuracy = train(file, train_loader, model, loss_function, optimizer, device)
        total_train_loss.append(train_loss) # Average Training Loss
        total_train_accuracy.append(train_accuracy) # Average Training Accuracy

        mid_time = time.time()
        training_time_difference = mid_time - start_time
        train_time_list.append(training_time_difference)

        test_val, test_accuracy = test(file,test_loader, model, loss_function, device)
        total_test_loss.append(test_val) # Average Test Loss
        total_test_accuracy.append(test_accuracy) # Average Test Accuracy

        end_time = time.time()
        test_time_difference = end_time - mid_time
        test_time_list.append(test_time_difference)
        total_time_difference = end_time - start_time
        total_time_list.append(total_time_difference)


    print("Training Loss:     ", total_train_loss)
    file.write("Training Loss:     " + str(total_train_loss) + "\n")
    print("Training Accuracy: ", total_train_accuracy)
    file.write("Training Accuracy: " + str(total_train_accuracy) + "\n")
    print("Testing Loss:      ", total_test_loss)
    file.write("Testing Loss:      " + str(total_test_loss) + "\n")
    print("Testing Accuracy:  ", total_test_accuracy)
    file.write("Testing Accuracy:  " + str(total_test_accuracy) + "\n")
    print("Training Time:     ", train_time_list)
    file.write("Training Time:     " + str(train_time_list) + "\n")
    print("Test Time:         ", test_time_list)
    file.write("Test Time:         " + str(test_time_list) + "\n")
    print("Total Time:        ", total_time_list)
    file.write("Total Time:        " + str(total_time_list) + "\n\n")

    print("Average Training Loss:     {}".format(np.mean(total_train_loss)))
    file.write("Average Training Loss:     {} \n".format(np.mean(total_train_loss)))
    print("Average Training Accuracy: {}".format(np.mean(total_train_accuracy)))
    file.write("Average Training Accuracy: {} \n".format(np.mean(total_train_accuracy)))
    print("Average Testing Loss:      {}".format(np.mean(total_test_loss)))
    file.write("Average Testing Loss:      {} \n".format(np.mean(total_test_loss)))
    print("Average Testing Accuracy:  {}".format(np.mean(total_test_accuracy)))
    file.write("Average Testing Accuracy:  {} \n".format(np.mean(total_test_accuracy)))
    print("Average Training Time:     {}".format(np.mean(train_time_list)))
    file.write("Average Training Time:     {} \n".format(np.mean(train_time_list)))
    print("Average Test Time:         {}".format(np.mean(test_time_list)))
    file.write("Average Test Time:         {}\n".format(np.mean(test_time_list)))
    print("Average Total Time:        {}".format(np.mean(total_time_list)))
    file.write("Average Total Time:        {} \n\n".format(np.mean(total_time_list)))

    # Model Evaluation
    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.legend()
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Training_Loss_vs_Training_Accuracy.png', bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_test_loss, label = 'Testing Loss')
    plt.plot(total_test_accuracy, label = 'Testing Accuracy')
    plt.legend()
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Testing_Loss_vs_Testing_Accuracy.png', bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.plot(total_test_loss, label = 'Testing Loss')
    plt.legend()
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Training_Loss_vs_Testing_Loss.png', bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.plot(total_test_accuracy, label = 'Testing Accuracy')
    plt.legend()
    if (save_fig):
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
    print("Confusion Matrix: \n", conf_mat, "\n")
    file.write("Confusion Matrix: \n" + str(conf_mat) + "\n\n")

    # plot the confusion matrix
    plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
    plot_confusion_matrix(conf_mat, plot_labels, save_fig)

    # Generate Classification Report
    report = classification_report(lbllist.numpy(), predlist.numpy(), target_names = plot_labels)
    print("Report: \n", report, "\n")
    file.write("Report: \n" + report + "\n")
