"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True
import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True


def tic():
    StartTime = time.time()
    return StartTime


def toc(StartTime):
    return time.time() - StartTime


def FindLatestModel(CheckPointPath):
    """
    Finds Latest Model in CheckPointPath
    Inputs:
    CheckPointPath - Path where you have stored checkpoints
    Outputs:
    LatestFile - File Name of the latest checkpoint
    """
    FileList = glob.glob(CheckPointPath + '*.ckpt') # * means all if need specific format then *.csv
    # print(FileList)
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    LatestFile = LatestFile.replace(CheckPointPath, '')
    LatestFile = LatestFile.replace('.ckpt.index', '')
    print(LatestFile)
    return LatestFile

