
#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      find_model.py
# Create:        04/12/22
# Description:   Pass back classifier model
#---------------------------------------------------------------------
# Pytorch libraries
from torch import nn
from torchvision import models

#---------------------------------------------------------------------
# Function:    findModel()
# Description: Select model and update last layer
#---------------------------------------------------------------------
def find_model(model_name, use_pretrained, num_classes, device):

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