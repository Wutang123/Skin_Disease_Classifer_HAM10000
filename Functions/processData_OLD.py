#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      process_Data.py
# Create:        01/17/22
# Description:   Function used to gather HAM1000 dataset and process data
#---------------------------------------------------------------------

# IMPORTS:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import itertools
import datetime
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    eda()
# Description: Exploratory Data Analysis.
#---------------------------------------------------------------------
def eda(skin_df, save_fig):

    # Plot count of 7 different cases
    fig = plt.figure()
    plt.title('Cell Type Count', fontsize = 15)
    skin_df['cell_type'].value_counts().plot.bar()
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Cell_Type_Count.png', bbox_inches = 'tight')

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
        fig.savefig('OUTPUT\Figures\DX&DX_Type_Count.png', bbox_inches = 'tight')

    # Plot Localization Count
    fig = plt.figure()
    plt.title('Localization Count', fontsize = 15)
    skin_df['localization'].value_counts().plot.bar()
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Localization_Count.png', bbox_inches = 'tight')

    # Plot Localization Histogram
    fig = plt.figure()
    plt.title('Age Histogram', fontsize = 15)
    skin_df['age'].hist(bins = 40)
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Age_Histogram.png', bbox_inches = 'tight')

    # Plot Sex Count
    fig = plt.figure()
    plt.title('Gender Count', fontsize = 15)
    skin_df['sex'].value_counts().plot.bar()
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Gender_Count.png', bbox_inches = 'tight')

    # Plot Age Distrubtion and Cell Type
    fig = plt.figure()
    plt.title('Age vs Cell Type', fontsize = 15)
    sns.scatterplot(x = 'age', y = 'cell_type', data = skin_df)
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Age_vs_Cell_Type_Scatter.png', bbox_inches = 'tight')

    # Plot Localization vs Gender
    fig = plt.figure(figsize=(25,10))
    plt.title('Localization vs Geneder', fontsize = 15)
    sns.countplot(y='localization', hue = 'sex', data = skin_df)
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Localization_vs_Gender.png', bbox_inches = 'tight')

    # Plot Localization vs Cell Type
    fig = plt.figure(figsize=(25,10))
    plt.title('Localization VS Cell Type',fontsize = 15)
    sns.countplot(y='localization', hue ='cell_type',data = skin_df)
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Localization_vs_Cell_Type.png', bbox_inches = 'tight')

    # Plot Age vs Cell Type
    fig = plt.figure(figsize=(25,10))
    plt.title('AGE VS CELL TYPE', fontsize = 15)
    sns.countplot(y='age', hue = 'cell_type', data = skin_df)
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Age_vs_Cell_Type.png', bbox_inches = 'tight')

    # Plot Gender vs Cell Type
    fig = plt.figure(figsize=(25,10))
    plt.title('GENDER VS CELL TYPE', fontsize = 15)
    sns.countplot(y='sex', hue = 'cell_type',data = skin_df)
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Gender_vs_Cell_Type.png', bbox_inches = 'tight')



#---------------------------------------------------------------------
# Function:    proccess_Data()
# Description: Function to process Data.
#---------------------------------------------------------------------
def model(input_shape, number_Cell_Type, save_fig, train_data, test_data):
    # CNN architechture: Input -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Output
    # train_data: [x_train, x_validate, y_train, y_validate]
    # test_data: [x_test, y_test]

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation ='relu', padding = 'Same',input_shape = input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation ='relu', padding = 'Same',))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation ='relu', padding = 'Same'))
    model.add(Conv2D(64, (3, 3), activation ='relu', padding = 'Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_Cell_Type, activation='softmax'))
    model.summary() # Display Model Summary

    # if(save_fig):
        # plot_model(model, to_file='OUTPUT\Figures\Model_Plot.png', show_shapes = True, show_layer_names = True) # Need pip install pydot and graphviz: https://graphviz.gitlab.io/download/

    # Define the optimizer
    optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)

    # Compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics = ["accuracy"])

    # Set a learning rate annealer
    learning_rate = ReduceLROnPlateau(monitor ='val_acc',
                                      patience =3,
                                      verbose =1,
                                      factor =0.5,
                                      min_lr =0.00001)

    # Data Augmentation to Prevent Overfitting - OPTIONAL STEP
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(train_data[0])

    # Train Model
    epochs = 50
    batch_size = 32
    history = model.fit(datagen.flow(train_data[0], train_data[2], batch_size = batch_size),
                                epochs = epochs, validation_data = (train_data[1], train_data[3]),
                                verbose = 1, steps_per_epoch = train_data[0].shape[0] // batch_size,
                                callbacks = [learning_rate])

    # Test Model
    loss, accuracy = model.evaluate(test_data[0], test_data[1], verbose=1)
    loss_v, accuracy_v = model.evaluate(train_data[1], train_data[3], verbose=1)
    print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
    print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

    current_time = datetime.datetime.now()
    model_path = "Models\model" + str(current_time) + ".h5"
    model.save(model_path)

#---------------------------------------------------------------------
# Function:    proccess_Data()
# Description: Function to process Data.
#---------------------------------------------------------------------
def proccess_Data(analysis_data):
    print(">>> START DATA PROCESS")

    save_fig = True
    input_shape = (100, 100, 3)
    base_data_dir = 'INPUT\HAM10000' # Location of base dataset directory

    # Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join(base_data_dir, '*', '*.jpg'))}
    # print(imageid_path_dict) # Verify images paths are found

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    number_Cell_Type = len(lesion_type_dict)

    skin_df = pd.read_csv(os.path.join(base_data_dir, 'HAM10000_metadata'))

    # Creating New Columns for better readability
    skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
    skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
    skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

    # print(skin_df.head()) # Print 1st 5 rows

    # print(skin_df.isnull().sum()) # Check if there is missing data.
    # skin_df['age'].fillna((skin_df['age'].mean()), inplace=True) # Fill in missing data with mean values (Only 57 'AGE' data cells are missing)

    # print(skin_df.dtypes) # Print data types of each column

    skin_df= skin_df[skin_df['age'] != 0] # Remove age = 0
    skin_df= skin_df[skin_df['age'] != 'unknown'] # Remove age = 'unknown'
    skin_df= skin_df[skin_df['sex'] != 'unknown'] # Remove sex = 'unknown'

    if(analysis_data):
        eda(skin_df, save_fig)

    skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,100))))

    # Display Sample (n = 5) of all Cell Types
    n_samples = 5
    fig, m_axs = plt.subplots(number_Cell_Type, n_samples, figsize = (4 * n_samples, 3 * number_Cell_Type))

    for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(['cell_type']).groupby('cell_type')):
        n_axs[0].set_title(type_name)
        for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state = 2018).iterrows()):
            c_ax.imshow(c_row['image'])
            c_ax.axis('off')
    if (save_fig):
        fig.savefig('OUTPUT\Figures\Image_Category_Samples.png')

    # Checking the Image Size
    # print(skin_df['image'].map(lambda x: x.shape).value_counts())

    features = skin_df.drop(columns=['cell_type_idx'], axis = 1) # Remove 'cell_type_idx' column
    # print(features)
    target = skin_df['cell_type_idx']
    # print(target)

    # Split Training and Test Data (80:20 Ratio)
    x_train_no_labels, x_test_no_labels, y_train_no_labels, y_test_no_labels = train_test_split(features, target, test_size = 0.20, train_size = 0.80, random_state = 1234)

    # Z-Score Normalization {x' = [(x - mean) / std_dev]}
    x_train = np.asarray(x_train_no_labels['image'].tolist())
    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)
    x_train = (x_train - x_train_mean)/x_train_std
    # print(x_train)

    x_test = np.asarray(x_test_no_labels['image'].tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean)/x_test_std
    # print(x_test)

    # Perform one-hot encoding on the labels
    y_train = to_categorical(y_train_no_labels, num_classes = number_Cell_Type)
    # print(y_train)
    y_test = to_categorical(y_test_no_labels, num_classes = number_Cell_Type)
    # print(y_test)

    # Split Training Dataset to Training and Validation (90:10 Ratio)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, train_size = 0.9, random_state = 2)

    # Reshape Image in 3 dimensions (Height = 100px, Width = 100px , Canal = 3)
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)
    x_validate = x_validate.reshape(x_validate.shape[0], *input_shape)

    # Build Model
    train_data = [x_train, x_validate, y_train, y_validate]
    test_data = [x_test, y_test]

    model(input_shape, number_Cell_Type, save_fig, train_data, test_data)


    print(">>> END DATA PROCESS")


#=====================================================================