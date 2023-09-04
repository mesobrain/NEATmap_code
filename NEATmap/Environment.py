# -*- coding: utf-8 -*-
"""
Environment
===========
"""
###############################################################################
### Python
###############################################################################
import os
import sys
import tifffile
import csv
import time
import tempfile
import itk
import copy
import re
import json
import shutil
import yaml
import tarfile
import h5py
import argparse
import logging
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import scipy.ndimage as ndi
import numpy as np
import SimpleITK as sitk
import scipy.stats as ss

###############################################################################
### NEATMap
###############################################################################
import Data_preprocessing.brain2d_to_3d as b2t3
import Data_preprocessing.cutting as cut
import Data_preprocessing.spot_segmentation as sps
import Data_preprocessing.tif2h5_slice as ths
import Data_preprocessing.tif2npz as tn

import Network.write_txt as wt
import Network.train as train
import Network.test as test
import Network.infer_whole_brain as infer_whole

import Splice_post.split as sl
import Splice_post.restore as rt
import Splice_post.post as post

import Registration.brain_registration as br

import Spot_mapping.seg3d_to_2d as s3t2
import Spot_mapping.Spot_trans as st

import Analysis.cortex_cell_counts as ct
import Analysis.cvandzscore as cz
import Analysis.Density as ds
import Analysis.freesia_export_to_BrainRegion as fb
import Analysis.get_region_layer as rl
import Analysis.Intensity as it
import Analysis.p_values as pv
import Analysis.Position_stats as ps
import Analysis.Statistics as stat
import Analysis.Volume_ratio as vr
import Analysis.log2Change as lc
import Analysis.spearman_corr as sc

__all__ = ['os', 'sys', 'tifffile', 'csv', 'time', 'tempfile', 'itk',
            'copy', 're', 'json', 'shutil', 'yaml', 'tarfile',
            'h5py', 'torch', 'argparse', 'logging', 'math',
            'warnings', 'nn', 'F', 'optim', 'pd', 'ndi', 'np', 
            'sitk', 'ss', 'b2t3', 'cut', 'sps', 'ths', 'tn',
            'wt', 'train', 'test', 'infer_whole', 'sl', 'rt',
            'post', 'br', 's3t2', 'st', 'ct', 'cz',
            'ds', 'fb', 'rl', 'it', 'pv', 'ps', 'stat', 
            'vr', 'lc', 'sc']