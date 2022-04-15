#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      processData.py
# Create:        01/17/22
# Description:   Function used to gather HAM1000 dataset and process data
#---------------------------------------------------------------------

# IMPORTS:
from Helper_Functions.dataset import *
from Helper_Functions.find_model import *
from Helper_Functions.find_dict_type import *

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import csv
from torchsummary import summary
from contextlib import redirect_stdout
import time

# pytorch libraries
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    train()
# Description: Train model
#---------------------------------------------------------------------
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

    train_loss /= num_batches
    train_accuracy /= size
    print(f"Training Error: [Accuracy: {(100 * train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f}]")
    file.write("Training Error: [Accuracy: {:>0.5f}%, Avg Loss: {:>8f}]\n".format((100 * train_accuracy), train_loss))

    return train_loss, train_accuracy



#---------------------------------------------------------------------
# Function:    val()
# Description: Validate model
#---------------------------------------------------------------------
def val(file, val_loader, model, loss_function, device):
    model.eval()

    size = len(val_loader.dataset)
    num_batches = len(val_loader)

    val_loss = 0
    val_accuracy = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Compute Prediction Error
            pred = model(images)
            loss = loss_function(pred, labels)
            val_loss += loss.item()
            val_accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()

    val_loss /= num_batches
    val_accuracy /= size
    print(f"Validate Error:  [Accuracy: {(100 * val_accuracy):>0.1f}%, Avg loss: {val_loss:>8f}]\n")
    file.write("Validate Error:  [Accuracy: {:>0.5f}%, Avg Loss: {:>8f}]\n".format((100 * val_accuracy), val_loss))

    return val_loss, val_accuracy



#---------------------------------------------------------------------
# Function:    calcResults()
# Description: Print and write all results
#---------------------------------------------------------------------
def calcResults(file, save_data, model_path, model_name, total_train_loss, total_train_accuracy, total_val_loss, total_val_accuracy, train_time_list, val_time_list, total_time_list):

    print("Training Loss:       ", total_train_loss)
    file.write("Training Loss:       " + str(total_train_loss) + "\n")
    print("Training Accuracy:   ", total_train_accuracy)
    file.write("Training Accuracy:   " + str(total_train_accuracy) + "\n")
    print("Validate Loss:       ", total_val_loss)
    file.write("Validate Loss:       " + str(total_val_loss) + "\n")
    print("Validate Accuracy:   ", total_val_accuracy)
    file.write("Validate Accuracy:   " + str(total_val_accuracy) + "\n")
    print("Training Time (sec): ", train_time_list)
    file.write("Training Time (sec): " + str(train_time_list) + "\n")
    print("Validate Time (sec): ", val_time_list)
    file.write("Validate Time (sec): " + str(val_time_list) + "\n")
    print("Total Time (sec):    ", total_time_list)
    file.write("Total Time (sec):    " + str(total_time_list) + "\n\n")
    print("")

    print("Best Training Accuracy:  ", max(total_train_accuracy), " at Epoch: ", total_train_accuracy.index(max(total_train_accuracy)))
    file.write("Best Training Accuracy:  " + str(max(total_train_accuracy)) + " at Epoch: " + str(total_train_accuracy.index(max(total_train_accuracy))) + "\n")
    print("Best Validate Accuracy:  ", max(total_val_accuracy), " at Epoch: ", total_val_accuracy.index(max(total_val_accuracy)))
    file.write("Best Validate Accuracy:  " + str(max(total_val_accuracy)) + " at Epoch: " + str(total_val_accuracy.index(max(total_val_accuracy))) + "\n\n")
    print("")

    print("Average Training Loss:       {}".format(np.mean(total_train_loss)))
    file.write("Average Training Loss:       {} \n".format(np.mean(total_train_loss)))
    print("Average Training Accuracy:   {}".format(np.mean(total_train_accuracy)))
    file.write("Average Training Accuracy:   {} \n".format(np.mean(total_train_accuracy)))
    print("Average Validate Loss:       {}".format(np.mean(total_val_loss)))
    file.write("Average Validate Loss:       {} \n".format(np.mean(total_val_loss)))
    print("Average Validate Accuracy:   {}".format(np.mean(total_val_accuracy)))
    file.write("Average Validate Accuracy:   {} \n".format(np.mean(total_val_accuracy)))
    print("Average Training Time (sec): {}".format(np.mean(train_time_list)))
    file.write("Average Training Time (sec): {} \n".format(np.mean(train_time_list)))
    print("Average Validate Time (sec): {}".format(np.mean(val_time_list)))
    file.write("Average Validate Time (sec): {}\n".format(np.mean(val_time_list)))
    print("Average Total Time (sec):    {}".format(np.mean(total_time_list)))
    file.write("Average Total Time (sec):    {} \n\n".format(np.mean(total_time_list)))
    print("")

    if(save_data):
        # Open csv file in 'w' mode
        with open(os.path.join(model_path, model_name + "_data.csv"), 'w', newline ='') as csv_file:
            length = len(total_train_loss)

            write = csv.writer(csv_file)
            write.writerow(["total_train_loss", "total_train_accuracy", "total_val_loss", "total_val_accuracy", "total_time_list"])
            for i in range(length):
                write.writerow([total_train_loss[i], total_train_accuracy[i], total_val_loss[i], total_val_accuracy[i], total_time_list[i]])


#---------------------------------------------------------------------
# Function:    plotFigures()
# Description: Plot comparsion plots
#---------------------------------------------------------------------
def plotFigures(save_fig, model_path, total_train_loss, total_train_accuracy, total_val_loss, total_val_accuracy):

    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(model_path, "Training_Loss_vs_Training_Accuracy.png")
        fig.savefig(save_path, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(total_val_loss, label = 'Validate Loss')
    plt.plot(total_val_accuracy, label = 'Validate Accuracy')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(model_path, "Validate_Loss_vs_Validate_Accuracy.png")
        fig.savefig(save_path, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.plot(total_val_loss, label = 'Validate Loss')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(model_path, "Training_Loss_vs_Validate_Loss.png")
        fig.savefig(save_path, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.plot(total_val_accuracy, label = 'Validate Accuracy')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(model_path, "Training_Accuracy_vs_Validate_Accuracy.png")
        fig.savefig(save_path, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)


#---------------------------------------------------------------------
# Function:    proccess_Data()
# Description: Function to process Data.
#---------------------------------------------------------------------
def helper_train(args, file, model_path, skin_df_train, skin_df_val, number_Cell_Type):

    save_fig       = args.save_fig
    save_model     = args.save_model
    save_data      = args.save_data
    use_pretrained = args.pretrained
    learning_rate  = args.lr
    batch          = args.batch
    num_worker     = args.worker
    epoch_num      = args.epoch
    input_size     = args.imgsz
    model_name     = args.model
    norm_mean      = (0.49139968, 0.48215827, 0.44653124)
    norm_std       = (0.24703233, 0.24348505, 0.26158768)
    num_classes    = number_Cell_Type

    print("Model:              {}".format(model_name))
    file.write("Model:              {} \n".format(model_name))
    print("Save Figures:       {}".format(save_fig))
    file.write("Save Figures:       {} \n".format(save_fig))
    print("Save Models:        {}".format(save_model))
    file.write("Save Models:        {} \n".format(save_model))
    print("Save Data:          {}".format(save_data))
    file.write("Save Data:          {} \n".format(save_data))
    print("Pretrained:         {}".format(use_pretrained))
    file.write("Pretrained:         {} \n".format(use_pretrained))
    print("Learning Rate:      {}".format(learning_rate))
    file.write("Learing Rate:       {} \n".format(learning_rate))
    print("Batch Size:         {}".format(batch))
    file.write("Batch Size:         {} \n".format(batch))
    print("Number of Workers:  {}".format(num_worker))
    file.write("Number of Workers:  {} \n".format(num_worker))
    print("Epoch Number:       {}".format(epoch_num))
    file.write("Epoch Number:       {} \n".format(epoch_num))
    print("Number of Classes:  {}".format(num_classes))
    file.write("Number of Classes:  {}".format(num_classes))
    print("Image Size:         {} by {}".format(input_size, input_size))
    file.write("Image Size:         {} by {} \n".format(input_size, input_size))
    print("Normalized Mean:   ", norm_mean)
    file.write("Normalized Mean:    " + str(norm_mean) + "\n")
    print("Normalized STD:    ", norm_std)
    file.write("Normalized STD:     " + str(norm_std) + "\n")

    # Define the device (use GPU if avaliable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Used:       ", device, "\n")
    file.write("Device Used:        " + str(device) + "\n")

    # Check to see if we load a trained model
    model = None
    model = find_model(model_name, use_pretrained, num_classes, device)

    # print("skin_df_train: ", skin_df_train)
    # print("skin_df_val: ", skin_df_val)

    # Dataset Transformations
    # Train
    train_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
                                        transforms.ColorJitter(brightness = 0.1, contrast = 0.1, hue = 0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    training_set = dataset(skin_df_train, transform = train_transform)
    train_loader = DataLoader(training_set, batch_size = batch, shuffle = True, num_workers = num_worker, drop_last = True)

    # Val
    val_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    val_set    = dataset(skin_df_val, transform = val_transform)
    val_loader = DataLoader(val_set, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = True)

    # Loss and Optimizer Function
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer    = optim.Adam(model.parameters(), lr = learning_rate)

    # Verify Tensor Size, should be [batch_size, channel_size, image_height, image_width] (e.g [32, 3, 225, 225])
    first_train = 1
    train_size = None
    for i, (images, labels) in enumerate(train_loader):
        if(first_train):
            train_size = images.shape
            first_train = 0
            # print(images.shape)
            # print(images.dtype)
            # print(images.device)
            # print(labels.shape)
            # print(labels.dtype)
            # print(labels.device)
        else:
            if(images.shape != train_size):
                print("ERROR: Mismatch train_loader Size!")
                file.write("ERROR: Mismatch train_loader Size!\n")
                sys.exit()

    first_val = 1
    val_size = None
    for i, (images, labels) in enumerate(val_loader):
        if(first_val):
            val_size = images.shape
            first_val = 0
        else:
            if(images.shape != val_size):
                print("ERROR: Mismatch val_loader Size!")
                file.write("ERROR: Mismatch val_loader Size!\n")
                sys.exit()

    # Summary of Model
    print("Tensor Image Size [batch_size, channel_size, image_height, image_width]: ", train_size)
    file.write("\nTensor Image Size [batch_size, channel_size, image_height, image_width]: " + str(train_size) + "\n\n")

    summary(model, input_size = (train_size[1], train_size[2], train_size[3]))
    with redirect_stdout(file):
        summary(model, input_size = (train_size[1], train_size[2], train_size[3]))

    # Main Training and Validate Function
    total_train_loss     = []
    total_train_accuracy = []
    total_val_loss       = []
    total_val_accuracy   = []
    train_time_list      = []
    val_time_list        = []
    total_time_list      = []

    for epoch in range(epoch_num):
        start_time = time.time()

        print("*" * 10)
        file.write("\n" + "*" * 10 +"\n")
        print("EPOCH {}:".format(epoch))
        file.write("EPOCH: " + str(epoch) + "\n")

        train_loss, train_accuracy = train(file, train_loader, model, loss_function, optimizer, device)
        total_train_loss.append(train_loss)         # Average Training Loss
        total_train_accuracy.append(train_accuracy) # Average Training Accuracy

        mid_time = time.time()
        training_time_difference = mid_time - start_time
        train_time_list.append(training_time_difference)

        val_loss, val_accuracy = val(file, val_loader, model, loss_function, device)
        total_val_loss.append(val_loss)         # Average Val Loss
        total_val_accuracy.append(val_accuracy) # Average Val Accuracy

        end_time = time.time()
        val_time_difference = end_time - mid_time
        val_time_list.append(val_time_difference)
        total_time_difference = end_time - start_time
        total_time_list.append(total_time_difference)

    print("*" * 10)
    file.write("*" * 10 +"\n")

    # Save Model
    if(save_model):
        save_path = os.path.join(model_path, "classifier.pth")
        torch.save(model.state_dict(), save_path)

    print("")
    file.write("\n")

    # Organize Results
    save_data = args.save_data
    calcResults(file, save_data, model_path, model_name, total_train_loss, total_train_accuracy, total_val_loss, total_val_accuracy, train_time_list, val_time_list, total_time_list)
    plotFigures(save_fig, model_path, total_train_loss, total_train_accuracy, total_val_loss, total_val_accuracy)
#=====================================================================