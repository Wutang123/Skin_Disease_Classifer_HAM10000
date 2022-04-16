#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      images2csv.py
# Create:        04/10/22
# Description:   Main function to create csv files
#---------------------------------------------------------------------

# IMPORTS:
import os
from datetime import datetime
import argparse
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import json

# sklearn libraries
from sklearn.model_selection import train_test_split

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    eda()
# Description: Exploratory Data Analysis
#---------------------------------------------------------------------
def eda(skin_df, save_fig, number_Cell_Type):

    skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,100))))

    # Plot count of 7 different cases
    fig = plt.figure()
    plt.title('Cell Type Count', fontsize = 15)
    skin_df['cell_type'].value_counts().plot.bar()
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Cell_Type_Count.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plotting of Technical Validation field (ground truth) which is dx_type to see the distribution of its 4 categories which are listed below :
    # 1. Histopathology(Histo): Histopathologic diagnoses of excised lesions have been performed by specialized dermatopathologists.
    # 2. Confocal: Reflectance confocal microscopy is an in-vivo imaging technique with a resolution at near-cellular level , and some facial benign with a grey-world assumption of all training-set images in Lab-color space before and after manual histogram changes.
    # 3. Follow-up: If nevi monitored by digital dermatoscopy did not show any changes during 3 follow-up visits or 1.5 years biologists accepted this as evidence of biologic benignity. Only nevi, but no other benign diagnoses were labeled with this type of ground-truth because dermatologists usually do not monitor dermatofibromas, seborrheic keratoses, or vascular lesions.
    # 4. Consensus: For typical benign cases without histopathology or followup biologists provide an expert-consensus rating of authors PT and HK. They applied the consensus label only if both authors independently gave the same unequivocal benign diagnosis. Lesions with this type of groundtruth were usually photographed for educational reasons and did not need further follow-up or biopsy for confirmation.
    fig = plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    plt.title('DX and DX_Type Count', fontsize = 15)
    skin_df['dx'].value_counts().plot.pie(autopct="%1.1f%%")
    plt.subplot(1,2,2)
    skin_df['dx_type'].value_counts().plot.pie(autopct="%1.1f%%")
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","DX&DX_Type_Count.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plot Localization Count
    fig = plt.figure()
    plt.title('Localization Count', fontsize = 15)
    skin_df['localization'].value_counts().plot.bar()
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Localization_Count.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plot Localization Histogram
    fig = plt.figure()
    plt.title('Age Histogram', fontsize = 15)
    skin_df['age'].hist(bins = 40)
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Age_Histogram.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plot Sex Count
    fig = plt.figure()
    plt.title('Gender Count', fontsize = 15)
    skin_df['sex'].value_counts().plot.bar()
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Gender_Count.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plot Age Distrubtion and Cell Type
    fig = plt.figure()
    plt.title('Age vs Cell Type', fontsize = 15)
    sns.scatterplot(x = 'age', y = 'cell_type', data = skin_df)
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Age_vs_Cell_Type_Scatter.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plot Localization vs Gender
    fig = plt.figure(figsize=(25,10))
    plt.title('Localization vs Geneder', fontsize = 15)
    sns.countplot(y='localization', hue = 'sex', data = skin_df)
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Localization_vs_Gender.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plot Localization vs Cell Type
    fig = plt.figure(figsize=(25,10))
    plt.title('Localization VS Cell Type',fontsize = 15)
    sns.countplot(y='localization', hue ='cell_type',data = skin_df)
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Localization_vs_Cell_Type.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plot Age vs Cell Type
    fig = plt.figure(figsize=(25,10))
    plt.title('AGE VS CELL TYPE', fontsize = 15)
    sns.countplot(y='age', hue = 'cell_type', data = skin_df)
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Age_vs_Cell_Type.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Plot Gender vs Cell Type
    fig = plt.figure(figsize=(25,10))
    plt.title('GENDER VS CELL TYPE', fontsize = 15)
    sns.countplot(y='sex', hue = 'cell_type',data = skin_df)
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Gender_vs_Cell_Type.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Display Sample (n = 5) of all Cell Types
    n_samples = 5
    fig, m_axs = plt.subplots(number_Cell_Type, n_samples, figsize = (4 * n_samples, 3 * number_Cell_Type))
    for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(['cell_type']).groupby('cell_type')):
        n_axs[0].set_title(type_name)
        for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state = 2018).iterrows()):
            c_ax.imshow(c_row['image'])
            c_ax.axis('off')
    if (save_fig):
        file_name = os.path.join("OUTPUT","Figures","Image_Category_Samples.png")
        fig.savefig(file_name, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():
    startT = time.time()
    parser = argparse.ArgumentParser(description = 'images2csv')
    parser.add_argument('--analysis'   , type = bool , default = False    , help = 'Conduct EDA')
    parser.add_argument('--save_fig'   , type = bool , default = True    , help = 'Save Figures')
    parser.add_argument('--save_data'  , type = bool , default = True    , help = 'Save Data to CSV')
    parser.add_argument('--input_path' , type = str  , default = 'INPUT' , help = 'Input path to data')

    args = parser.parse_args()

    print(">>>>> images2csv \n")

    dir_path = os.path.join(args.input_path, "HAM10000")
    print("Input Dataset Path: ", dir_path)

    # Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join(dir_path, '*', '*.jpg'))}

    lesion_type_dict = {
        'akiec': 'Actinic keratoses',              # cell_type_idx: 0
        'bcc'  : 'Basal cell carcinoma',           # cell_type_idx: 1
        'bkl'  : 'Benign keratosis-like lesions ', # cell_type_idx: 2
        'df'   : 'Dermatofibroma',                 # cell_type_idx: 3
        'nv'   : 'Melanocytic nevi',               # cell_type_idx: 4
        'mel'  : 'Melanoma',                       # cell_type_idx: 5
        'vasc' : 'Vascular lesions'                # cell_type_idx: 6
    }

    skin_df = pd.read_csv(os.path.join(dir_path, 'HAM10000_metadata'))

    # Creating New Columns for better readability
    skin_df['path']          = skin_df['image_id'].map(imageid_path_dict.get)
    skin_df['cell_type']     = skin_df['dx'].map(lesion_type_dict.get)
    skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

    number_Cell_Type = (skin_df['cell_type_idx'].max() + 1) # Find total number of cell types (7)

    skin_df= skin_df[skin_df['age'] != 0]         # Remove age = 0
    skin_df= skin_df[skin_df['age'] != 'unknown'] # Remove age = 'unknown'
    skin_df= skin_df[skin_df['sex'] != 'unknown'] # Remove sex = 'unknown'

    # Sort and drop duplicates
    skin_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

    if(args.analysis):
            print("Conducting EDA...\n")
            eda(skin_df, args.save_fig, number_Cell_Type)
            print("Completed EDA...\n")

    # Split Dataset into Training and Test Set
    skin_df_train, skin_df_test = train_test_split(skin_df, test_size = 0.3)
    skin_df_val, skin_df_test = train_test_split(skin_df_test, test_size = 0.5)

    skin_df_train = skin_df_train.reset_index()
    skin_df_val   = skin_df_val.reset_index()
    skin_df_test  = skin_df_test.reset_index()

    print("Training Dataset Count: ")
    print(skin_df_train)
    print(skin_df_train['cell_type'].value_counts().sort_index(), "\n")
    print("Validation Dataset Count: ")
    print(skin_df_val)
    print(skin_df_val['cell_type'].value_counts().sort_index(), "\n")
    print("Testing Dataset Count: ")
    print(skin_df_test)
    print(skin_df_test['cell_type'].value_counts().sort_index(), "\n\n")

    if(args.save_data):
        skin_df_train.to_csv(os.path.join(args.input_path, "Train_Classifier_Dataset.csv"))
        skin_df_val.to_csv(os.path.join(args.input_path,   "Val_Classifier_Dataset.csv"))
        skin_df_test.to_csv(os.path.join(args.input_path,  "Test_Classifier_Dataset.csv"))

    endT = time.time()
    program_time_difference = endT - startT
    min = math.floor(program_time_difference / 60)
    sec = math.floor(program_time_difference % 60)
    print("Total Program Time (min:sec): " + str(min) + ":" + str(sec))

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================