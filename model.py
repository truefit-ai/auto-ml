from .metrics import *
from .vision_model import *
from .lgb_model import *
from .xformer_model import *
from .segmentation_model import *

import os
import sys


hour = 3600

ACCEL = None
if ACCEL is not None: print(' ** {}x FASTER RUN **'.format(ACCEL))
else: ACCEL = 1

from types import SimpleNamespace   

import random
import datetime
import math
import numpy as np
import os
import time

import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
    
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler

    
import timm


class Model:
    def __init__(self, metadata):
        self.metadata_ = metadata
        
    def train(self, dataset, val_dataset = None, val_metadata = None,
                          remaining_time_budget = 5 * hour, seed = None):

        remaining_time_budget /= ACCEL
        self.remaining_time_budget = remaining_time_budget
        
        self.start_time = time.time()
        self.end_time = self.start_time + 0.8 * self.remaining_time_budget
                            
        self.train_dataset = dataset 
        self.val_dataset = val_dataset
        
        metric = dataset.metadata.get_final_metric()
        task_type = dataset.metadata.get_task_type()
        print(task_type, metric)
        
        features = []; targets = []
        idxs = random.sample( list(np.arange(len(dataset))),
                          k = min(5000, len(dataset)))
        for i in idxs:
            features.append(dataset[i][0])
            targets.append(dataset[i][1])
            
        input_shape = np.median( np.stack([e.shape for e in features]), 
                                axis = 0 ).astype(int)#round(0)
        output_shape = np.median( np.stack([e.shape for e in targets]), 
                                axis = 0 ).astype(int)#round(0)
        sorted_input_shape = sorted(input_shape)
        print(len(dataset))
        print(input_shape)
        print(sorted_input_shape)
        
        print(output_shape)
        
        max_ratio = sorted_input_shape[-1] / sorted_input_shape[-2]
        print('max_ratio: {:.1f}'.format(max_ratio))
        
        if len(output_shape) > 1:
            self.model_class = SegmentationModel        
            
            
        # ONLY ONE AXIS --> lgb or transformer
        elif sorted_input_shape[-2] == 1: 
            feature_var = np.stack([f.flatten() for f in features]).std(axis = 0)
            var_ratio = feature_var.std() / np.median(feature_var)
            print('variance ratio: {:.1f}'.format(var_ratio))
            
            if var_ratio > 10 and task_type in ['single-label', 'continous']:       
                self.model_class = LGBModel
            else:
                self.model_class = XformerModel
                
        # 2d/wide
        elif max_ratio > 2.5 and sorted_input_shape[-3] == 1: # wide 2-d:
            self.model_class = XformerModel
            
        # else, vision:
        else:
            self.model_class = VisionModel
        
        if ( metric == 'zero_one_error'
                and task_type == 'single-label'
                    and tuple(sorted_input_shape) == (1, 1, 16, 52)  ):
            print(" NINAPRO CAPPED AT 0.5 HOURS !!!")
            remaining_time_budget = min(0.5 * hour, remaining_time_budget)
            
        
        print(); print(self.model_class); print()
        self.model = self.model_class(self.metadata_)
        self.model.train(dataset, 
                               val_dataset, val_metadata,
                          remaining_time_budget, seed)
        
        
                
        print('Train Time: {:.1f} sec. ({:.0%} of allotted)'.format(
                time.time() - self.start_time, 
                (time.time() - self.start_time) /  self.remaining_time_budget ))
        
        
    def test(self, dataset, remaining_time_budget=None):
        yp = self.model.test(dataset, remaining_time_budget)
        print('Inference Time: {:.1f} sec. ({:.0%} of allotted)'.format(
                time.time() - self.start_time, 
                (time.time() - self.start_time) /  self.remaining_time_budget ))
        return yp
