#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      test_classifier.py
# Create:        01/17/22
# Description:   Main test classifier functions
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
    parser.add_argument('--jetson'    , type = bool , default = True                                                         , help = 'Currently using Jetson')
    parser.add_argument('--batch'     , type = int  , default = 2                                                            , help = 'Select Batch Size (e.g 32)')
    parser.add_argument('--worker'    , type = int  , default = 1                                                            , help = 'Select Number of Workers (e.g 4)')
    parser.add_argument('--imgsz'     , type = int  , default = 64                                                           , help = 'Select Input Image Size (e.g 225)')
    parser.add_argument('--test_csv'  , type = str  , default = 'Test_Classifier_Dataset.csv'                                , help = 'Load testing csv files')
    parser.add_argument('--model_path', type = str  , default = 'Run0'                                                       , help = 'Load model path')
    parser.add_argument('--model_name', type = str  , default = 'alexnet'                                                    , help = 'model_name')
    # parser.add_argument('--model_name', type = str  , default = 'efficientnet_b0'                                            , help = 'model_name')
    # parser.add_argument('--model_name', type = str  , default = 'mobilenet_v2'                                               , help = 'model_name')
    # parser.add_argument('--model_name', type = str  , default = 'resnet50'                                                   , help = 'model_name')
    # parser.add_argument('--model_name', type = str  , default = 'shufflenet_v2_x1_0'                                         , help = 'model_name')
    # parser.add_argument('--model_name', type = str  , default = 'squeezenet1_1'                                              , help = 'model_name')
    # parser.add_argument('--model_name', type = str  , default = 'vgg16'                                                      , help = 'model_name')

    args = parser.parse_args()

    print(">>>>> HAM10000 Test Classifer \n")
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H.%M.%S")

    model_name      = args.model_name
    model_file_path = os.path.join("OUTPUT", "Models", model_name, "Train", args.model_path, "classifier.pth")
    jetson_logfile  = os.path.join("OUTPUT", "Models", model_name, "Test", args.model_path, "jetson_stat_log.csv")

    cont = True
    count = 0
    model_path = os.path.join("OUTPUT", "Models", model_name, "Test", "Run" + str(count))
    while cont:
        if(os.path.isdir(model_path)):
            count += 1
            model_path = os.path.join("OUTPUT", "Models", model_name, "Test", "Run" + str(count))
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

    test_csv = os.path.join("INPUT", args.test_csv)
    skin_df_test = pd.read_csv(test_csv, index_col = 0)
    skin_df_test = skin_df_test.reset_index()
    number_Cell_Type = 7

    lesion_id_dict, lesion_type_dict, colors_dict = find_dict_type(file)
    helper_test(args, file, model_path, model_file_path, model_name, jetson_logfile, skin_df_test, number_Cell_Type, lesion_id_dict, lesion_type_dict, colors_dict)

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