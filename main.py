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

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():
    print("=====> START MAIN FUNCTION")

    analysis_data = False # Set to True if you want to conduct Exploratory Data Analysis (EDA)
    proccess_Data(analysis_data)

    print("=====> END MAIN FUNCTION")

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================