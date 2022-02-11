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
from Functions.processData import *
import os
from datetime import datetime
import argparse
import math

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():
    startT = time.time()
    parser = argparse.ArgumentParser(description = 'HAM10000 Classifer')
    parser.add_argument('--analysis'  , type = bool , default = False, help = 'Conduct EDA')
    parser.add_argument('--save_fig'  , type = bool , default = True , help = 'Save Figures')
    parser.add_argument('--save_model', type = bool , default = True , help = 'Save Model')
    parser.add_argument('--save_data' , type = bool , default = True , help = 'Save Data to CSV')
    parser.add_argument('--pretrained', type = bool , default = True , help = 'Use Pretrained Models (e.g True)')
    parser.add_argument('--lr'        , type = float, default = 1e-4 , help = 'Select Learning Rate (e.g. 1e-3)')
    parser.add_argument('--epoch'     , type = int  , default = 50   , help = 'Select Epoch Size (e.g 50)')
    parser.add_argument('--batch'     , type = int  , default = 32   , help = 'Select Batch Size (e.g 32)')
    parser.add_argument('--worker'    , type = int  , default = 4    , help = 'Select Number of Workers (e.g 4)')
    parser.add_argument('--imgsz'     , type = int  , default = 225  , help = 'Select Input Image Size (e.g 225)')
    parser.add_argument('--load'      , type = str  , default = ''   , help = 'Load Model')
    parser.add_argument('--model'     , type = str  , default = 'vgg16'
                                      , help = 'alexnet, efficientnet_b0, mobilenet_v2, resnet50, shufflenet_v2_x1_0, squeezenet1_1, vgg16')

    args = parser.parse_args()

    print(">>>>> Starting Program \n")
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H.%M.%S")

    cont = True
    count = 0
    model_path = os.path.join("OUTPUT\Models", args.model, "Run" + str(count))
    while cont:
        if(os.path.isdir(model_path)):
            count += 1
            model_path = os.path.join("OUTPUT\Models", args.model, "Run" + str(count))
        else:
            os.mkdir(model_path)
            cont = False

    log_file = os.path.join(model_path, "log_file.txt")

    file = open(log_file, "a")
    file.write("=" * 10 + "\n")
    file.write("Log File Generated On: "+ date_time + "\n")
    file.write("-" * 10 + "\n")
    file.write(str(args) + "\n")

    analysis_data = args.analysis
    save_fig      = args.save_fig
    save_model    = args.save_model
    proccess_Data(args, file, analysis_data, save_fig, save_model, model_path)

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