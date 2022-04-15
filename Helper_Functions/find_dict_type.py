#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      find_dict_type.py
# Create:        04/12/22
# Description:   Pass dictionaries
#---------------------------------------------------------------------

# IMPORTS:
# Python libraries
import json

#---------------------------------------------------------------------
# Function:    find_dict_type()
# Description: Plot confusion matrix
#---------------------------------------------------------------------
def find_dict_type(file):

    lesion_id_dict = {
        0 : 'Actinic keratoses',              # cell_type_idx: 0
        1 : 'Basal cell carcinoma',           # cell_type_idx: 1
        2 : 'Benign keratosis-like lesions ', # cell_type_idx: 2
        3 : 'Dermatofibroma',                 # cell_type_idx: 3
        4 : 'Melanocytic nevi',               # cell_type_idx: 4
        5 : 'Melanoma',                       # cell_type_idx: 5
        6 : 'Vascular lesions'                # cell_type_idx: 6
    }
    file.write("lesion_id_dict: \n")
    file.write(json.dumps(lesion_id_dict))
    file.write("\n\n")

    lesion_type_dict = {
        'akiec': 'Actinic keratoses',              # cell_type_idx: 0
        'bcc'  : 'Basal cell carcinoma',           # cell_type_idx: 1
        'bkl'  : 'Benign keratosis-like lesions ', # cell_type_idx: 2
        'df'   : 'Dermatofibroma',                 # cell_type_idx: 3
        'nv'   : 'Melanocytic nevi',               # cell_type_idx: 4
        'mel'  : 'Melanoma',                       # cell_type_idx: 5
        'vasc' : 'Vascular lesions'                # cell_type_idx: 6
    }
    file.write("lesion_type_dict: \n")
    file.write(json.dumps(lesion_type_dict))
    file.write("\n\n")

    # Color for each class (used in scatter plot)
    colors_dict = {0 : 'tab:blue'  ,
                    1 : 'tab:orange',
                    2 : 'tab:green' ,
                    3 : 'tab:red'   ,
                    4 : 'tab:purple',
                    5 : 'tab:pink'  ,
                    6 : 'tab:cyan'  }
    file.write("colors_dict: \n")
    file.write(json.dumps(colors_dict))
    file.write("\n\n")

    return lesion_id_dict, lesion_type_dict, colors_dict