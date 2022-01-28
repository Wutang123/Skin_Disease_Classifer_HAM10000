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
import logging
import glob

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():


    #  TODO: REMOVE LATER
    remove_dir = glob.glob('OUTPUT\Log\*')
    for f in remove_dir:
        os.remove(f)




    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H.%M.%S")
    log_file = "OUTPUT\Log\log_file_" + date_time + ".txt"

    file = open(log_file, "a")
    file.write("=" * 10 + "\n")
    file.write("Log File Generated On: "+ date_time + "\n")
    file.write("-" * 10 + "\n")
    file.write("=====> STARTING MAIN FUNCTION \n")
    print("=====> STARTING MAIN FUNCTION")

    analysis_data = False # Set to True if you want to conduct Exploratory Data Analysis (EDA)
    proccess_Data(file, analysis_data)

    file.write("=====> ENDING MAIN FUNCTION \n")
    file.write("=" * 10)
    print("=====> ENDING MAIN FUNCTION \n")
    file.close()

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================