#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      main.py
# Create:        01/17/22
# Description:   Main function to call other functions
#---------------------------------------------------------------------

# IMPORTS:
from Helper_Functions.helper_test import *

import time
import os
from datetime import datetime
import argparse
import math
import pandas as pd

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():
    startT = time.time()
    parser = argparse.ArgumentParser(description = 'HAM10000 Test Classifer')
    parser.add_argument('--save_fig'  , type = bool , default = True                                                         , help = 'Save Figures')
    parser.add_argument('--save_data' , type = bool , default = True                                                         , help = 'Save Data to CSV')
    parser.add_argument('--pretrained', type = bool , default = True                                                         , help = 'Use Pretrained Models (e.g True)')
    parser.add_argument('--batch'     , type = int  , default = 32                                                           , help = 'Select Batch Size (e.g 32)')
    parser.add_argument('--worker'    , type = int  , default = 4                                                            , help = 'Select Number of Workers (e.g 4)')
    parser.add_argument('--imgsz'     , type = int  , default = 225                                                          , help = 'Select Input Image Size (e.g 225)')
    parser.add_argument('--test_csv'  , type = str  , default = 'Input/Test_Classifier_Dataset.csv'                          , help = 'Load testing csv files')
    parser.add_argument('--model_path', type = str  , default = 'OUTPUT/Models/alexnet/Train/Run0/classifier.pth'            , help = 'Load model path')
    # parser.add_argument('--model_path', type = str  , default = 'OUTPUT/Models/efficientnet_b0/Train/Run0/classifier.pth'    , help = 'Load model path')
    # parser.add_argument('--model_path', type = str  , default = 'OUTPUT/Models/mobilenet_v2/Train/Run0/classifier.pth'       , help = 'Load model path')
    # parser.add_argument('--model_path', type = str  , default = 'OUTPUT/Models/resnet50/Train/Run0/classifier.pth'           , help = 'Load model path')
    # parser.add_argument('--model_path', type = str  , default = 'OUTPUT/Models/shufflenet_v2_x1_0/Train/Run0/classifier.pth' , help = 'Load model path')
    # parser.add_argument('--model_path', type = str  , default = 'OUTPUT/Models/squeezenet1_1/Train/Run0/classifier.pth'      , help = 'Load model path')
    # parser.add_argument('--model_path', type = str  , default = 'OUTPUT/Models/vgg16/Train/Run0/classifier.pth'              , help = 'Load model path')

    args = parser.parse_args()

    print(">>>>> HAM10000 Test Classifer \n")
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H.%M.%S")

    model_file_path = args.model_path
    model_name      = model_file_path.split("/")
    model_name      = model_name[2]

    cont = True
    count = 0
    model_path = os.path.join("OUTPUT\Models", model_name, "Test\Run" + str(count))
    while cont:
        if(os.path.isdir(model_path)):
            count += 1
            model_path = os.path.join("OUTPUT\Models", model_name, "Test\Run" + str(count))
        else:
            os.mkdir(model_path)
            cont = False

    log_file = os.path.join(model_path, "log_file.txt")

    file = open(log_file, "a")
    file.write("=" * 10 + "\n")
    file.write("Log File Generated On: "+ date_time + "\n")
    file.write("-" * 10 + "\n")
    print(args,"\n")
    file.write(str(args) + "\n")

    skin_df_test = pd.read_csv(args.test_csv, index_col = 0)
    skin_df_test = skin_df_test.reset_index()
    number_Cell_Type = 7

    lesion_id_dict, lesion_type_dict, colors_dict = find_dict_type(file)
    helper_test(args, file, model_path, model_file_path, model_name, skin_df_test, number_Cell_Type, lesion_id_dict, lesion_type_dict, colors_dict)

    endT = time.time()
    program_time_difference = endT - startT
    min = math.floor(program_time_difference/60)
    sec = math.floor(program_time_difference%60)
    print("")
    print("Total Program Time (min:sec): " + str(min) + ":" + str(sec))
    file.write("\n")
    file.write("Total Program Time (min:sec): " + str(min) + ":" + str(sec) + "\n")

    file.write("=" * 10)
    file.close()

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================