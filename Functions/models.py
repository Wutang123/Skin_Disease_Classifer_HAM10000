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
# Description:   Train, Validate, and Test dataset on models
#---------------------------------------------------------------------

# IMPORTS:
# Python libraries
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from glob import glob
from PIL import Image
import time
import seaborn as sns
import csv

# sklearn libraries
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, plot_roc_curve, accuracy_score, auc

# pytorch libraries
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Class:
#---------------------------------------------------------------------
# Function:    Dataset()
# Description: Characterizes HAM10000 for PyTorch
#---------------------------------------------------------------------
class Dataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, df, transform = None):
        'Initialization'
        self.df = df
        self.transform = transform

    def __len__(self):
        # Denotes the total number of samples
        return len(self.df)

    def __getitem__(self, index):
        # Generates one sample of data
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y




# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    findModel()
# Description: Select model and update last layer
#---------------------------------------------------------------------
def findModel(model_name, use_pretrained, num_classes, device):
    # Models: alexnet, vgg16, resnet50, squeezenet1_1, shufflenet_v2_x1_0, mobilenet_v2, efficientnet_b0

    model = None
    num_ftrs = 0

    if model_name == "alexnet":
        model = models.alexnet(pretrained = use_pretrained)
        num_ftrs = model.classifier[6].in_features # Find number of features in last layer
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained = use_pretrained)
        num_ftrs = model.classifier[6].in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained = use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained = use_pretrained)
        model.fc = nn.Conv2d(512, num_classes, kernel_size = (1,1), stride = (1,1))

    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained = use_pretrained)
        num_ftrs = model.fc.in_features

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained = use_pretrained)
        num_ftrs = model.classifier[1].in_features
        model.fc = nn.Linear(num_ftrs,num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained = use_pretrained)
        num_ftrs = model.classifier[1].in_features
        model.fc = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid Model Name. Exiting...")
        exit()

    model = model.to(device) # Add device to model

    return model



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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
# Function:    plot_confusion_matrix()
# Description: Plot confusion matrix
#---------------------------------------------------------------------
def plot_confusion_matrix(file, target_names, save_fig, model_path, y_label, y_pred):

    conf_mat = confusion_matrix(y_label, y_pred)
    print("Confusion Matrix: \n", conf_mat, "\n")
    file.write("Confusion Matrix: \n" + str(conf_mat) + "\n\n")

    fig = plt.figure()
    plt.title('Confusion matrix')
    ax = sns.heatmap(conf_mat, annot = True, cmap = plt.cm.Blues, fmt = 'g', linewidths = 1)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set(ylabel="True Labels", xlabel="Predicted Labels")

    # Drawing the frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if (save_fig):
        save_path = os.path.join(model_path, "Confusion_Matrix.png")
        fig.savefig(save_path, bbox_inches = 'tight')

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(1)
    for i in range(len(target_names)):
        print(target_names[i], ": ", round(class_accuracy[i],2), "%")
        file.write(target_names[i] + "=" + str(round(class_accuracy[i],2)) + "% \n")
    file.write("\n\n")


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

    print("Best Training Accuracy: ", max(total_train_accuracy), " at Epoch: ", total_train_accuracy.index(max(total_train_accuracy)))
    file.write("Best Training Accuracy: " + str(max(total_train_accuracy)) + " at Epoch: " + str(total_train_accuracy.index(max(total_train_accuracy))) + "\n")
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

    fig = plt.figure()
    plt.plot(total_val_loss, label = 'Validate Loss')
    plt.plot(total_val_accuracy, label = 'Validate Accuracy')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(model_path, "Validate_Loss_vs_Validate_Accuracy.png")
        fig.savefig(save_path, bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.plot(total_val_loss, label = 'Validate Loss')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(model_path, "Training_Loss_vs_Validate_Loss.png")
        fig.savefig(save_path, bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.plot(total_val_accuracy, label = 'Validate Accuracy')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(model_path, "Training_Accuracy_vs_Validate_Accuracy.png")
        fig.savefig(save_path, bbox_inches = 'tight')



#---------------------------------------------------------------------
# Function:    genReport()
# Description: Generate confusion matrix, classification report, and AUC ROC score
#---------------------------------------------------------------------
def genReport(model, file, save_fig, test_loader, model_path, device, num_classes):

    model.eval()
    y_label    = torch.zeros(0, dtype = torch.long, device = 'cpu')
    y_pred     = torch.zeros(0, dtype = torch.long, device = 'cpu')
    y_pred_auc = torch.zeros(0, dtype = torch.long, device = 'cpu')

    with torch.no_grad():
        correctly_identified = 0
        total_images = 0
        for images, labels in test_loader:
            N = images.size(0)
            images   = images.to(device)
            labels   = labels.to(device)
            outputs  = model(images)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            y_label    = torch.cat([y_label, labels.view(-1).cpu()])
            y_pred     = torch.cat([y_pred, preds.view(-1).cpu()])
            y_pred_auc = torch.cat([y_pred_auc, nn.functional.softmax(outputs, dim = 1).cpu()])

            for i in range(N):
                soft_max_output = nn.functional.softmax(outputs[i], dim = -1)
                max_index = torch.argmax(soft_max_output)
                total_images += 1
                correctly_identified += int(labels[i] == max_index)
        print("Correctly identified = ", correctly_identified, " Total_images = ", total_images, " Accuracy = ", (float(correctly_identified)/total_images) * 100, "\n")
        file.write("Correctly identified = " + str(correctly_identified) + " Total_images = " + str(total_images) + " Accuracy = " + str((float(correctly_identified)/total_images) * 100) + "\n\n")

    y_label = y_label.numpy()
    y_pred = y_pred.numpy()
    y_pred_auc = y_pred_auc.numpy()

    target_names = ['AKIEC','BCC','BKL','DF','NV','MEL','VASC']

    # Confusion Matrix
    plot_confusion_matrix(file, target_names, save_fig, model_path, y_label, y_pred)

    # Classification Report
    report = classification_report(y_label, y_pred, target_names = target_names)
    print("Report: \n", report, "\n")
    file.write("Report: \n" + report + "\n")



#---------------------------------------------------------------------
# Function:    model()
# Description: Train and Validate dataset on multiple models
#---------------------------------------------------------------------
def model(args, file, save_fig, save_model, skin_df_train, skin_df_val, skin_df_test, number_Cell_Type, model_path):

    learning_rate = args.lr
    batch = args.batch
    num_worker = args.worker
    epoch_num = args.epoch
    # epoch_num = 3
    input_size = args.imgsz
    use_pretrained = args.pretrained
    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)
    num_classes = number_Cell_Type

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
    file.write("Device Used:        " + str(device) + "\n")

    # Check to see if we load a trained model
    model = None
    model_name = args.model
    model = findModel(model_name, use_pretrained, num_classes, device)
    if args.load:
        model.load_state_dict(torch.load(args.load))

    # Dataset Transformations
    # Train
    train_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
                                        transforms.ColorJitter(brightness = 0.1, contrast = 0.1, hue = 0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    training_set = Dataset(skin_df_train, transform = train_transform)
    train_loader = DataLoader(training_set, batch_size = batch, shuffle = True, num_workers = num_worker)

    # Val
    val_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    val_set    = Dataset(skin_df_val, transform = val_transform)
    val_loader = DataLoader(val_set, batch_size = batch, shuffle = False, num_workers = num_worker)

    # Test
    test_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    test_set    = Dataset(skin_df_test, transform = test_transform)
    test_loader = DataLoader(test_set, batch_size = batch, shuffle = False, num_workers = num_worker)

    # Loss and Optimizer Function
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer    = optim.Adam(model.parameters(), lr = learning_rate)

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
        total_train_loss.append(train_loss) # Average Training Loss
        total_train_accuracy.append(train_accuracy) # Average Training Accuracy

        mid_time = time.time()
        training_time_difference = mid_time - start_time
        train_time_list.append(training_time_difference)

        val_loss, val_accuracy = val(file, val_loader, model, loss_function, device)
        total_val_loss.append(val_loss) # Average Val Loss
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
        save_path = os.path.join(model_path, model_name + ".pth")
        torch.save(model.state_dict(), save_path)

    print("")
    file.write("\n")

    # Organize Results
    save_data = args.save_data
    calcResults(file, save_data, model_path, model_name, total_train_loss, total_train_accuracy, total_val_loss, total_val_accuracy, train_time_list, val_time_list, total_time_list)
    plotFigures(save_fig, model_path, total_train_loss, total_train_accuracy, total_val_loss, total_val_accuracy)
    genReport(model, file, save_fig, test_loader, model_path, device, num_classes)
