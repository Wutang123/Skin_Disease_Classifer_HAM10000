#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      helper_test.py
# Create:        01/22/22
# Description:   Helper function used to test classifier
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
import seaborn as sns
import csv
import pandas as pd
from jtop import jtop, JtopException

# sklearn libraries
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.manifold import TSNE

# pytorch libraries
import torch
from torch import  nn
from torch.utils.data import DataLoader
from torchvision import transforms

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    plot_confusion_matrix()
# Description: Plot confusion matrix
#---------------------------------------------------------------------
def plot_confusion_matrix(file, target_names, save_fig, model_path, y_label, y_pred, run_images_path):

    conf_mat = confusion_matrix(y_label, y_pred)
    print("Confusion Matrix: \n", conf_mat, "\n")
    file.write("Confusion Matrix: \n" + str(conf_mat) + "\n\n")

    fig = plt.figure()
    plt.title('Confusion matrix')
    ax = sns.heatmap(conf_mat, annot = True, cmap = plt.cm.Blues, fmt = 'g', linewidths = 1)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set(ylabel = "True Labels", xlabel = "Predicted Labels")

    # Drawing the frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if (save_fig):
        save_path = os.path.join(run_images_path, "Confusion_Matrix.png")
        fig.savefig(save_path, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(1)
    for i in range(len(target_names)):
        print(target_names[i], ": ", round(class_accuracy[i],2), "%")
        file.write(target_names[i] + "=" + str(round(class_accuracy[i],2)) + "% \n")

    print()
    file.write("\n")



#---------------------------------------------------------------------
# Function:    plot_roc_auc()
# Description: Plot AUC and AUC-ROC score
#---------------------------------------------------------------------
def plot_roc_auc(file, save_data, save_fig, model_path, run_images_path, y_label, y_pred, y_pred_auc, y_true, lesion_id_dict):

    # Plot all labels
    fig_all = plt.figure(num = 0, figsize = (10, 10))
    plt.title("ROC_AUC_All")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    indices = []
    for key, value in lesion_id_dict.items():

        # Plot each label
        fig = plt.figure(num = 1, figsize = (10, 10))
        plt.title("ROC_AUC_" + str(value))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        for i, label in enumerate(y_label):
            if(label == key):
                indices.append(i)

        # Extract the indexes in y_label and y_pred
        current_y_label    = np.take(y_label, indices)
        current_y_pred     = np.take(y_pred, indices)
        current_y_pred_auc = np.take(y_pred_auc, indices)
        current_y_true     = np.take(y_true, indices)

        # ROC Curve
        fpr, tpr, _ = roc_curve(current_y_true, current_y_pred_auc)
        if(save_data):
            fpr_length   = len(fpr)

            with open(os.path.join(model_path, "test_" + str(value) + "_roc_curve_data.csv"), 'w', newline ='') as csv_file:
                write = csv.writer(csv_file)
                write.writerow(["false_positive_rate", "true_positive_rate"])
                for i in range(fpr_length):
                    write.writerow([fpr[i], tpr[i]])

        # AUC ROC Score
        auc = 0
        try:
            auc = roc_auc_score(current_y_true, current_y_pred_auc)
        except ValueError as e:
            print("Try_catch Error: ", e)
            file.write("Try_catch Error: {}\n".format(e))
            pass

        print("AUC Score ({}): {}".format(str(value), auc))
        file.write("AUC Score ({}): {}\n".format(str(value), auc))

        fig_all = plt.figure(num = 0, figsize = (10, 10))
        plt.plot(fpr, tpr, label = (str(value) + "_AUC = " + str(auc)))
        plt.legend(loc='best')

        fig = plt.figure(num = 1, figsize = (10, 10))
        plt.plot(fpr, tpr, label = (str(value) + "AUC = " + str(auc)))
        plt.legend(loc='best')

        if(save_fig):
                save_path = os.path.join(run_images_path, "ROC_AUC_" + str(value) + ".png")
                fig.savefig(save_path, bbox_inches = 'tight')
        fig.clf()
        plt.close(fig)
        indices.clear() # Clear indices list

    if(save_fig):
        save_path = os.path.join(run_images_path, "ROC_AUC_All.png")
        fig_all.savefig(save_path, bbox_inches = 'tight')
    fig_all.clf()
    plt.close(fig_all)



#---------------------------------------------------------------------
# Function:    visual_tsne()
# Description: Visualize TSNE
#---------------------------------------------------------------------
def visual_tsne(file, save_data, save_fig, model_path, run_images_path, features, preds, labels, lesion_id_dict, name, colors_dict):

    # Constants
    N       = 2
    PERP    = [25.0]
    LR      = 'auto'
    ITER    = [500]
    VERBOSE = 1
    STATE   = 0

    for i in PERP:
        for j in ITER:
            print("n_components = {}, perplexity = {}, learning_rate = {}, n_iter = {}, verbose = {}, random_state = {}".format(N, i, LR, j, VERBOSE, STATE))
            file.write("n_components = {}, perplexity = {}, learning_rate = {}, n_iter = {}, verbose = {}, random_state = {} \n".format(N, i, LR, j, VERBOSE, STATE))
            tsne = TSNE(n_components = N, perplexity = i,  learning_rate = LR, n_iter = j,  verbose = VERBOSE, random_state = STATE).fit_transform(features)

            print("tsne:", tsne)
            print("tsne.shape: ", tsne.shape, "\n")
            file.write("tsne.shape: {} \n".format(tsne.shape))

            if(save_data):
                tsne_data = pd.DataFrame(tsne)
                tsne_data.columns = ["x_coord", "y_coord"]
                tsne_data["labels"] = labels
                tsne_data.to_csv(os.path.join(model_path, name + "_" + str(i) + "_" + str(j) + "_tsne_data.csv"))

            # Extract x and y coordinates representing the positions of the images on T-SNE plot
            tsne_x = tsne[:, 0]
            tsne_y = tsne[:, 1]

            # Plot all labels
            fig_all = plt.figure(num = 0, figsize = (10, 10))
            plt.title(name + '_All_TSNE_Labeled')
            plt.xlabel('t-SNE-1')
            plt.ylabel('t-SNE-2')

            indices = []
            for key, value in lesion_id_dict.items():

                # Plot each label
                fig = plt.figure(num = 1, figsize = (10, 10))
                plt.title(name + '_' + str(value) + '_TSNE_Labeled')
                plt.xlabel('t-SNE-1')
                plt.ylabel('t-SNE-2')

                # Find all indices for each class
                for i, pred in enumerate(preds):
                    if(pred == key):
                        indices.append(i)

                # Extract the coordinates of the points of this class only
                current_tsne_x = np.take(tsne_x, indices)
                current_tsne_y = np.take(tsne_y, indices)

                fig_all = plt.figure(num = 0, figsize = (10, 10))
                plt.scatter(current_tsne_x, current_tsne_y, c = colors_dict.get(key), label = value)
                plt.legend(loc='best')

                fig = plt.figure(num = 1, figsize = (10, 10))
                plt.scatter(current_tsne_x, current_tsne_y, c = colors_dict.get(key), label = value)
                plt.legend(loc='best')

                if(save_fig):
                    save_path = os.path.join(run_images_path, name + "_" + str(i) + "_" + str(j) + "_" + value + "_TSNE_Labeled.png")
                    fig.savefig(save_path, bbox_inches = 'tight')
                fig.clf()
                plt.close(fig)
                indices.clear() # Clear indices list

            if(save_fig):
                plt.legend(loc='best')
                save_path = os.path.join(run_images_path, name + "_" + str(i) + "_" + str(j) + "_All_TSNE_Labeled.png")
                fig_all.savefig(save_path, bbox_inches = 'tight')
            fig_all.clf()
            plt.close(fig_all)

        file.write("\n")

    return tsne_data



#---------------------------------------------------------------------
# Function:    plot_tsne()
# Description: Plot addition t-SNE plots
#---------------------------------------------------------------------
def plot_tsne(run_images_path, save_fig, original, y_output, lesion_id_dict, num_classes):

    print("Plotting addition t-SNE plots...")

    # Overlay Plots - All Class
    fig_0 = plt.figure(num = 0, figsize = (10, 10))
    plt.title('Overlay_All_TSNE_Labeled')
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    sns.scatterplot(data = original, x = "x_coord", y = "y_coord", hue = "labels", palette = sns.color_palette(n_colors = num_classes))
    sns.scatterplot(data = y_output, x = "x_coord", y = "y_coord", hue = "labels", palette = sns.color_palette("husl", n_colors = num_classes))
    plt.legend(labels = ["0_original", "1_original", "2_original",
                         "3_original", "4_original", "5_original",
                         "6_original", "1_y_output", "2_y_output",
                         "3_y_output", "4_y_output", "5_y_output",
                         "6_y_output"],
               loc='best')

    # Side-by-side Plots - All Class
    fig_1 = plt.figure(num = 1, figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('SideBySide_All_original_Labeled')
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.legend(labels = ["1_original", "2_original", "5_original",
                         "6_original", "9_original", "11_original",
                         "12_original", "13_original", "14_original",],
               loc='best')
    sns.scatterplot(data = original, x = "x_coord", y = "y_coord", hue = "labels", palette = sns.color_palette(n_colors = num_classes))
    plt.subplot(1, 2, 2)
    plt.title('SideBySide_All_y_output_Labeled')
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.legend(labels = ["1_y_output", "2_y_output", "5_y_output",
                         "6_y_output", "9_y_output", "11_y_output",
                         "12_y_output", "13_y_output", "14_y_output"],
               loc='best')
    sns.scatterplot(data = y_output, x = "x_coord", y = "y_coord", hue = "labels", palette = sns.color_palette("husl", n_colors = num_classes))

    if(save_fig):
        save_path_0 = os.path.join(run_images_path, "Overlay_All_TSNE_Labeled.png")
        save_path_1 = os.path.join(run_images_path, "SideBySide_All_TSNE_Labeled.png")
        fig_0.savefig(save_path_0, bbox_inches = 'tight')
        fig_1.savefig(save_path_1, bbox_inches = 'tight')

    fig_0.clf()
    fig_1.clf()

    for key, value in lesion_id_dict.items():

        original_contain_values = (original['labels'] == key)
        y_output_contain_values = (y_output['labels'] == key)

        # Overlay Plots - Each Class
        fig_2 = plt.figure(num = 2, figsize=(10, 10))
        plt.title("Overlay_" + str(value) + "_TSNE_Labeled")
        plt.xlabel('t-SNE-1')
        plt.ylabel('t-SNE-2')
        plt.scatter(original.loc[original_contain_values, 'x_coord'], original.loc[original_contain_values, 'y_coord'], c = 'tab:cyan', label = str(key) + '_original')
        plt.scatter(y_output.loc[y_output_contain_values, 'x_coord'], y_output.loc[y_output_contain_values, 'y_coord'], c = 'tab:pink', label = str(key) + '_y_output')
        plt.legend(loc='best')

        # Side-by-side Plots - Each Class
        fig_3 = plt.figure(num = 3, figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title("SideBySide_" + str(value) + "_original_Labeled", fontsize = 15)
        plt.xlabel('t-SNE-1')
        plt.ylabel('t-SNE-2')
        plt.scatter(original.loc[original_contain_values, 'x_coord'], original.loc[original_contain_values, 'y_coord'], c = 'tab:cyan', label = str(key) + '_original')
        plt.legend(loc='best')
        plt.subplot(1, 2, 2)
        plt.title("SideBySide_" + str(value) + "_y_output_Labeled", fontsize = 15)
        plt.xlabel('t-SNE-1')
        plt.ylabel('t-SNE-2')
        plt.scatter(y_output.loc[y_output_contain_values, 'x_coord'], y_output.loc[y_output_contain_values, 'y_coord'], c = 'tab:pink', label = str(key) + '_y_output')
        plt.legend(loc='best')

        if(save_fig):
            save_path_0 = os.path.join(run_images_path, "Overlay_" + str(value) + "_TSNE_Labeled.png")
            save_path_1 = os.path.join(run_images_path, "SideBySide_" + str(value) + "_TSNE_Labeled.png")
            fig_2.savefig(save_path_0, bbox_inches = 'tight')
            fig_3.savefig(save_path_1, bbox_inches = 'tight')

        fig_2.clf()
        fig_3.clf()

    plt.close(fig_0)
    plt.close(fig_1)
    plt.close(fig_2)
    plt.close(fig_3)

    print("Completed t-SNE plots...\n")



#---------------------------------------------------------------------
# Function:    helper_test()
# Description: Helper function used to test classifier
#---------------------------------------------------------------------
def helper_test(args, file, model_path, model_file_path, model_name, jetson_logfile, skin_df_test, number_Cell_Type, lesion_id_dict, lesion_type_dict, colors_dict):

    run_images_path = os.path.join(model_path, "run_images")
    os.mkdir(run_images_path)

    save_fig        = args.save_fig
    save_data       = args.save_data
    use_pretrained  = args.pretrained
    batch           = args.batch
    num_worker      = args.worker
    input_size      = args.imgsz
    input_size      = args.imgsz
    jetson          = args.jetson
    norm_mean       = (0.49139968, 0.48215827, 0.44653124)
    norm_std        = (0.24703233, 0.24348505, 0.26158768)
    num_classes     = number_Cell_Type

    print("Model File Path:    {}".format(model_file_path))
    file.write("Model File Path:    {} \n".format(model_file_path))
    print("Model:              {}".format(model_name))
    file.write("Model:              {} \n".format(model_name))
    if(jetson):
        print("Jeston File Path:   {}".format(jetson_logfile))
        file.write("Jeston File Path:   {} \n".format(jetson_logfile))
    print("Save Figures:       {}".format(save_fig))
    file.write("Save Figures:       {} \n".format(save_fig))
    print("Save Data:          {}".format(save_data))
    file.write("Save Data:          {} \n".format(save_data))
    print("Pretrained:         {}".format(use_pretrained))
    file.write("Pretrained:         {} \n".format(use_pretrained))
    print("Batch Size:         {}".format(batch))
    file.write("Batch Size:         {} \n".format(batch))
    print("Number of Workers:  {}".format(num_worker))
    file.write("Number of Workers:  {} \n".format(num_worker))
    print("Number of Classes:  {}".format(num_classes))
    file.write("Number of Classes:  {}".format(num_classes))
    print("Jetson:             {}".format(jetson))
    file.write("Jetson:             {}".format(jetson))
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
    model.load_state_dict(torch.load(model_file_path))  # Load model

    # print("skin_df_test: ", skin_df_test)

    # Dataset Transformations
    # Test
    test_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    test_set    = dataset(skin_df_test, transform = test_transform)
    test_loader = DataLoader(test_set, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = True)

    # Verify Tensor Size, should be [batch_size, channel_size, image_height, image_width] (e.g [32, 3, 225, 225])
    if(not jetson):
        first_test = 1
        test_size = None
        for i, (images, labels) in enumerate(test_loader):
            if(first_test):
                test_size = images.shape
                first_test = 0
            else:
                if(images.shape != test_size):
                    print("ERROR: Mismatch test_loader Size!")
                    file.write("ERROR: Mismatch test_loader Size!\n")
                    sys.exit()

    model.eval()

    # Collect Inference Time
    starter, ender  = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    batch_time_list = []
    total_images    = 0

    # Collect Jetson Stats
    jetson_stats = []

    # Ground truth of image
    original   = torch.zeros(0, dtype = torch.long, device = 'cpu')

    # Output of classifier
    y_output   = torch.zeros(0, dtype = torch.long, device = 'cpu')
    y_prob     = torch.zeros(0, dtype = torch.long, device = 'cpu')
    y_label    = torch.zeros(0, dtype = torch.long, device = 'cpu')
    y_pred     = torch.zeros(0, dtype = torch.long, device = 'cpu')
    y_pred_auc = torch.zeros(0, dtype = torch.long, device = 'cpu')

    with torch.no_grad():
        correctly_identified = 0
        total_images = 0
        for batch, (images, labels) in enumerate(test_loader):

            if (batch == 0) or (batch % 25 == 0):
                print("Batch: ", batch)
                print("*" * 5)

                with jtop() as jetson_jtop:
                    stats = jetson_jtop.stats
                    jetson_stats.append(stats)

            images_per_batch = images.size(0)
            images   = images.to(device)
            labels   = labels.to(device)
            starter.record()
            outputs  = model(images)
            ender.record()
            _, preds = torch.max(outputs, 1)
            # print(outputs, outputs.shape)

            # Get prediction probability
            prob = nn.functional.softmax(outputs, dim=1)
            top_p, top_class = prob.topk(1, dim = 1)
            # print(prob, prob.shape, "\n")
            # print(top_p, top_p.shape, "\n")

            # Wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            batch_time_list.append(curr_time)

            # Append batch prediction results
            original   = torch.cat([original,   images.view(images.size(0), -1).cpu()])
            y_output   = torch.cat([y_output,   outputs.cpu()])
            y_prob     = torch.cat([y_prob,     prob.cpu()])
            y_label    = torch.cat([y_label,    labels.view(-1).cpu()])
            y_pred     = torch.cat([y_pred,     preds.view(-1).cpu()])
            y_pred_auc = torch.cat([y_pred_auc, top_p.view(-1).cpu()])

            for i in range(images_per_batch):
                soft_max_output = nn.functional.softmax(outputs[i], dim = -1)
                max_index = torch.argmax(soft_max_output)
                total_images += 1
                correctly_identified += int(labels[i] == max_index)

            # break

        print()
        print("Correctly identified = ", correctly_identified, " Total_images = ", total_images, " Accuracy = ", (float(correctly_identified)/total_images) * 100, "\n")
        file.write("Correctly identified = " + str(correctly_identified) + " Total_images = " + str(total_images) + " Accuracy = " + str((float(correctly_identified)/total_images) * 100) + "\n")

    total_batch_time = np.sum(batch_time_list)
    mean_batch_time  = total_batch_time / total_images
    std_batch_time   = np.std(total_batch_time)

    file.write("\n")
    print("total_images:              {}".format(total_images))
    file.write("total_images:              {}\n".format(total_images))
    print("total_inference_time(sec): {}".format(total_batch_time))
    file.write("total_inference_time(sec): {}\n".format(total_batch_time))
    print("mean_inference_time(sec):  {}".format(mean_batch_time))
    file.write("mean_inference_time(sec):  {}\n".format(mean_batch_time))
    print("std_inference_time(sec):   {}".format(std_batch_time))
    file.write("std_inference_time(sec):   {}\n".format(std_batch_time))
    print()
    file.write("\n")

    original   = original.numpy()
    y_output   = y_output.numpy()
    y_prob     = y_prob.numpy()
    y_label    = y_label.numpy()
    y_pred     = y_pred.numpy()
    y_pred_auc = y_pred_auc.numpy()

    # Write jetson stats to csv
    if(jetson):
        jetson_df = pd.DataFrame(jetson_stats)
        jetson_df.to_csv(jetson_logfile)

    target_names = ['AKIEC','BCC','BKL','DF','NV','MEL','VASC']

    # Determine if it was true or false
    y_true  = []
    for i, label in enumerate(y_label):
            if(label == y_pred[i]):
                y_true.append(1) # True
            else:
                y_true.append(0) # False

    # Save CSV File
    if(save_data):
        test_image = skin_df_test['image_id']
        y_length   = len(y_label)

        with open(os.path.join(model_path, "test_classifier_GT_data.csv"), 'w', newline ='') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(["image", "ground_truth", "predict", "prob", "F_or_T"])
            for i in range(y_length):
                write.writerow([test_image[i], lesion_id_dict[y_label[i]], lesion_id_dict[y_pred[i]], y_pred_auc[i], y_true[i]])

    # Confusion Matrix
    plot_confusion_matrix(file, target_names, save_fig, model_path, y_label, y_pred, run_images_path)

    # Classification Report
    report = classification_report(y_label, y_pred, target_names = target_names)
    print("Report: \n", report)
    file.write("Report: \n" + report + "\n")

    # ROC and ROC-AUC Score
    plot_roc_auc(file, save_data, save_fig, model_path, run_images_path, y_label, y_pred, y_pred_auc, y_true, lesion_id_dict)

    # T-SNE
    if(not jetson):
        original_tsne = visual_tsne(file, save_data, save_fig, model_path, run_images_path, original, y_pred, y_label, lesion_id_dict, 'original', colors_dict)
        y_output_tsne = visual_tsne(file, save_data, save_fig, model_path, run_images_path, y_output, y_pred, y_label, lesion_id_dict, 'y_output', colors_dict)
        plot_tsne(run_images_path, save_fig, original_tsne, y_output_tsne, lesion_id_dict, num_classes)
