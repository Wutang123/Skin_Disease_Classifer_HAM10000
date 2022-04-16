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
import os
from jtop import jtop, JtopException
import csv
from datetime import datetime
import argparse

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description = 'Jetson Stats Logger')
    parser.add_argument('--model', type = str, default = "alexnet"           , help = "Model name")
    parser.add_argument('--model', type = str, default = "efficientnet_b0"   , help = "Model name")
    parser.add_argument('--model', type = str, default = "mobilenet_v2"      , help = "Model name")
    parser.add_argument('--model', type = str, default = "resnet50"          , help = "Model name")
    parser.add_argument('--model', type = str, default = "shufflenet_v2_x1_0", help = "Model name")
    parser.add_argument('--model', type = str, default = "squeezenet1_1"     , help = "Model name")
    parser.add_argument('--model', type = str, default = "vgg16"             , help = "Model name")
    args = parser.parse_args()

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H.%M.%S")

    jetson_logfile = os.path.join("OUTPUT\Models", args.model, "Test", "jetson_stat_log_" + date_time + ".csv")

    # Collect jetson stats while running classifier test
    with jtop() as jetson:
            with open(jetson_logfile, 'w') as csvfile:
                stats = jetson.stats
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                writer.writeheader()
                writer.writerow(stats)
                while jetson.ok():
                    stats = jetson.stats
                    writer.writerow(stats)

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================