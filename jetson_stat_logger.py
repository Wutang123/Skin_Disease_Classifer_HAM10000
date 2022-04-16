#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Haiyan Hu
# Name:          Justin Wu
# Project:       Skin Disease Classifier
# Function:      jetson_stat_logger.py
# Create:        04/16/22
# Description:   Function used to collect jetson stat data
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
    # parser.add_argument('--model', type = str, default = "efficientnet_b0"   , help = "Model name")
    # parser.add_argument('--model', type = str, default = "mobilenet_v2"      , help = "Model name")
    # parser.add_argument('--model', type = str, default = "resnet50"          , help = "Model name")
    # parser.add_argument('--model', type = str, default = "shufflenet_v2_x1_0", help = "Model name")
    # parser.add_argument('--model', type = str, default = "squeezenet1_1"     , help = "Model name")
    # parser.add_argument('--model', type = str, default = "vgg16"             , help = "Model name")
    args = parser.parse_args()

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H.%M.%S")

    jetson_logfile = os.path.join("OUTPUT", "Models", args.model, "Test", "jetson_stat_log_" + date_time + ".csv")
    print(jetson_logfile)

    # Collect jetson stats while running classifier test
    try:
        with jtop() as jetson:
            with open(jetson_logfile, 'w') as csvfile:
                stats = jetson.stats
                writer = csv.DictWriter(csvfile, fieldnames = stats.keys())
                writer.writeheader()
                writer.writerow(stats)
                while jetson.ok():
                    stats = jetson.stats
                    writer.writerow(stats)
    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed with CTRL-C")
    except IOError:
        print("I/O error")

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================