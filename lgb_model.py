from .metrics import *
from .lgb import *

import sys
import logging


ACCEL = None
if ACCEL is not None: print(' * {}x FASTER RUN *'.format(ACCEL))

# try: import lightgbm as lgb; print('lgb')
# except: pass;

import datetime
import logging
import numpy as np
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

from sklearn.metrics import zero_one_loss, f1_score, log_loss

import torch
import numpy as np

hour = 3600


from sklearn.model_selection import ParameterSampler



def merge_batches(dataloader: DataLoader, is_single_label: bool):    
    x_batches = []
    y_batches = []
    for x, y in dataloader:
        x = x.detach().numpy()
        x = x.reshape(x.shape[0], -1)
        x_batches.append(x)
        
        y = y.detach().numpy()
        if len(y.shape) > 2:
            y = y.reshape(y.shape[0], -1)
        
        if is_single_label: 
            y = np.argmax(y, axis=1)
            
        y_batches.append(y)
    
    x_matrix = np.concatenate(x_batches, axis = 0)
    y_matrix = np.concatenate(y_batches, axis = 0)
    
    return x_matrix, y_matrix


class LGBModel:
    def __init__(self, metadata):
        self.metadata_ = metadata
        # self.task = self.metadata_.get_dataset_name()
        self.task_type = self.metadata_.get_task_type()
        print('task_type:', self.task_type)
        
        self.final_metric = self.metadata_.get_final_metric()
        print('final metric:', self.final_metric)
        
        self.metric = get_metric_func(self.final_metric, self.task_type)
        print('metric:', self.metric)
        
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_size = math.prod(self.metadata_.get_output_shape())

        self.num_examples_train = self.metadata_.size()
        self.num_train = self.metadata_.size()
        self.num_test = self.metadata_.get_output_shape()

        # Getting the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device Found =', self.device)
        
        self.output_shape = self.metadata_.get_output_shape()
        self.input_shape = self.metadata_.get_tensor_shape()
        print('\nINPUT SHAPE =', self.input_shape)
        print('\nOUTPUT SHAPE =', self.output_shape)
        
        self.train_batch_size = 64
        self.test_batch_size = 64

    def get_dataloader(self, dataset, batch_size, split):
        if split == "train":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle = True,
                drop_last = False,
                collate_fn = dataset.collate_fn,
                # num_workers = os.cpu_count()
            )
        elif split == "test":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle = False,
                collate_fn = dataset.collate_fn,
                # num_workers = os.cpu_count()
            )
        return dataloader

    def get_model(self, params):
        return get_lgb_model(params, self.task_type,
                             self.output_size, self.final_metric)

    def fitAndPredict(self, model, x_train, y_train):
        model.fit(x_train, y_train,)
        y_val = self.y_val
        yp_val = getattr(model, 'predict_proba'
                                 if hasattr(model, 'predict_proba')
                             else 'predict')( self.x_val )
        try: 
            if len(yp_val.shape) > len(y_val.shape):
                y_val = OHE(y_val, yp_val.shape[1])
            score = self.metric(y_val, yp_val)
        except Exception as e: print(e); score = 1.1111; 
        
        return model, score
              
    def train(self, dataset, val_dataset = None, val_metadata = None,
                          remaining_time_budget = 5 * hour, seed = None):
        
        if ACCEL: remaining_time_budget /= ACCEL;        
        
        self.scope(dataset, val_dataset, val_metadata, remaining_time_budget,
                      seed or datetime.datetime.now().microsecond)
        self.trials()
        while not self.done_training and not self.done_trialing:
            self.trial()
        self.fit();
        
            
    def scope(self, dataset, val_dataset = None, val_metadata = None,
                          remaining_time_budget = 5 * hour, seed = 1,
                             max_allocation = 1.,
             ):
        self.done_training = False
        self.done_trialing = False
        self.remaining_time_budget = remaining_time_budget
        self.start_time = time.time()
        self.end_time = (self.start_time + self.remaining_time_budget * 0.8)
        self.max_allocation = max_allocation
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.val_metadata = val_metadata
        self.seed = seed
        if not hasattr(self, "trainloader"):
            self.trainloader = self.get_dataloader(
                dataset, 
                self.train_batch_size,
                'train',
            )

        # Initialize Data - Flat Sets for LGB
        self.x_train, self.y_train = merge_batches(self.trainloader, 
                                         self.task_type == 'single-label' )
        print(self.x_train.shape, self.y_train.shape)
        
        self.x_full_train, self.y_full_train = self.x_train, self.y_train

        if self.val_dataset:
            self.valloader = self.get_dataloader(self.val_dataset, 
                                    self.test_batch_size, 'test')
            self.x_val, self.y_val = merge_batches(self.valloader,
                                        self.task_type == 'single-label' )
        else:
            self.idx_train, self.idx_val, _, _ = train_test_split(
                 *[np.arange(len(self.dataset))] * 2,
                    random_state = seed if seed is not None
                        else datetime.datetime.now().microsecond)
            
            self.x_train, self.x_val, self.y_train, self.y_val = (
                self.x_train[self.idx_train], self.x_train[self.idx_val], 
                    self.y_train[self.idx_train], self.y_train[self.idx_val])
        
        print('{:.1f}s data loading, {:.0f}s remain\n'.format(
            time.time() - self.start_time,
            self.end_time - time.time()  
        ))

        
        cpu_mult = max(1, min(4, os.cpu_count() // 8))
        train_size = self.x_full_train.size
        num_rows = self.x_full_train.shape[0]
        num_cols = self.x_full_train.shape[1]
        cpu_time = (self.remaining_time_budget 
                             * cpu_mult * self.max_allocation)
        
        print('targeting a cpu time of {}s'.format(cpu_time))
              
        self.lgb_params = get_base_lgb_params(cpu_time, num_rows, num_cols)
        print(self.lgb_params); print()
        
        min_train_start = time.time()
        model = self.get_model(self.lgb_params)
        model, score = self.fitAndPredict(model, 
                                        self.x_train, self.y_train,)
        self.fast_model_time = time.time() - min_train_start
        self.model = model
        self.n_trained = 1
        self.models = []
        print('Min-Model Score: {:.4f} in {:.0f}s'.format(
                        score, self.fast_model_time))
        
        ratio = (self.end_time - time.time() ) / (self.fast_model_time ) 
        print('\nexpecting time to train {:.1f}x as many models'.format(ratio)
                 + ', {:.0f}s remain\n'.format(self.end_time - time.time()))
        
        # Expanded Model
        if ratio > 15:
            self.lgb_params = rescale_lgb_params(self.lgb_params, 
                                                 ratio ** 0.95 / 7)
            print(self.lgb_params); print();
            
            fast_model_start = time.time()
            model = self.get_model(self.lgb_params)       
            model, score = self.fitAndPredict(model, self.x_train, self.y_train)
            self.model = model
            print('Fast-Model Score: {:.4f} in {:.0f}s'.format(
                        score, time.time() - fast_model_start))
            self.fast_model_time = time.time() - fast_model_start
            self.n_trained += 1

            ratio = (self.end_time - time.time() 
                        ) / self.fast_model_time 

            print('\nexpecting time to train {:.1f}x as many models'.format(ratio)
                + ', {:.0f}s remain\n'.format(self.end_time - time.time()))
        
        if ratio < max_allocation: 
            print('  returning as-is');
            self.done_training = True
            return
        
    def trials(self, n_trials = 5):
        if self.done_training: return;
        self.experiments = []
    
        ratio = (self.end_time - time.time() ) / (self.fast_model_time ) 
        if ratio < (5 if self.n_trained == 1 else 5): 
            print(' not enough time for trials');
            self.done_trialing = True; return;
    
        self.n_trials = n_trials
        ratio = (self.end_time - time.time() ) / self.fast_model_time 
        
        
        ratio = ratio / (2 * self.n_trials)
        print('expanding model size by {:.2f}x\n'.format(ratio))
        self.lgb_param_dict = get_lgb_param_dict(self.lgb_params, ratio)
    
    def _trial_check(self):
        if self.done_training: return;
        if len(self.experiments) > 0:
            best = sorted(self.experiments, key = lambda x: x[1])[0]
            ratio_remain = (self.end_time - time.time()
                        ) / (best[-1] * len(self.x_full_train) / len(self.x_train))
            if ratio_remain < max(2, len(self.experiments) ): 
                print('trials complete')
                self.done_trialing = True
                
            elif len(self.experiments) >= 12: 
                print('plenty of experiments\n'); 
                self.done_trialing = True

            
    def trial(self):
        self._trial_check();
        if self.done_training or self.done_trialing: return;
        params = list(ParameterSampler(self.lgb_param_dict, 1, 
                     random_state = datetime.datetime.now().microsecond))[0]
        print(params);
        model_start = time.time()
        model = self.get_model(params)       
        model, score = self.fitAndPredict(model, self.x_train, self.y_train)

        t = time.time() - model_start
        self.experiments.append((params, score, t))
        print(' trial scored {:.3f} in {:.0f}s\n'.format(score, t))
        
        
    def blind_fit(self, allotted_time = None):
        allotted_time = allotted_time or (self.end_time - time.time()) 
        ratio = ( allotted_time
                      / ( self.fast_model_time *
                             (len(self.x_full_train) / len(self.x_train)) ) )
        print(' training one final model with {:.1f}x scale'.format(ratio))
        
        self.lgb_params = get_final_lgb_params(self.lgb_params, ratio)
        print(self.lgb_params); print();

        final_model_start = time.time()
        model = self.get_model(self.lgb_params)     
        model.fit(self.x_full_train, self.y_full_train)
        self.model = model
        self.done_training = True
        print('Final-Model Score: {:.4f} in {:.0f}s'.format(
                    np.nan, time.time() - final_model_start))
        print(
            "Total time: {:.1f} sec. ({:.0%} of allotted)".format(
                time.time() - self.start_time, 
                (time.time() - self.start_time) /  self.remaining_time_budget
            )
        )

    def fit(self, allotted_time = None):
        allotted_time = allotted_time or (self.end_time - time.time()) 
        
        if len(self.experiments) < 3: 
            print('too few experiments'); 
            self.blind_fit();
            return
        
        best = sorted(self.experiments, key = lambda x: x[1])
        n_ensemble = 2 if len(self.experiments) >= 9 else 1
        
        print(' training {} final models'.format(n_ensemble)
                + ', {:.0f}s remain\n'.format(self.end_time - time.time()))
        
        ensemble_weights = np.arange(1., 1. + n_ensemble)[::-1] ** 1.25
        self.models = []
        for i in range(n_ensemble):
            lgb_params = best[i][0]
            t = best[i][-1]
            ratio = (self.end_time - time.time()
                    ) / (t * len(self.x_full_train) / len(self.x_train)
                    ) * (ensemble_weights[i] / ensemble_weights[i:].sum())
            print('\nexpecting time to train {:.1f}x longer\n'.format(ratio))
            if i > 0 and ratio < 5: print('   skipping extra iterations'); break;
            if ratio > 10:
                ratio = 10
                print('   capping ratio at {}x'.format(ratio));

            lgb_params = get_full_lgb_params(lgb_params, ratio)
            print(lgb_params)
            final_model_start = time.time()
            model = self.get_model(lgb_params)
            model.fit(self.x_full_train, self.y_full_train,)
            if i == 0: self.model = model
            self.models.append(model)
            print('Final-Model {} Score: {:.4f} in {:.0f}s'.format(
                        i, np.nan, time.time() - final_model_start))         

        self.done_training = True;
        print(
            "Total time used for training: {:.1f} sec. ({:.0%} of allotted)".format(
                time.time() - self.start_time, 
                (time.time() - self.start_time) /  self.remaining_time_budget
            )
        )


    def test(self, dataset, remaining_time_budget=None):
        test_begin = time.time()

        print("Begin testing...")
        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )
        
        x_test, _ = merge_batches(self.testloader,
                                  (self.task_type == 'single-label') )
        
        # get test predictions from the model
        preds = []
        models = self.models if len(self.models) > 0 else [self.model]
        ensemble_weights = np.arange(1., 1 + len(models))[::-1] ** 1.25
        ensemble_weights /= ensemble_weights.sum()
        print(ensemble_weights)
        for i, model in enumerate(models):
            preds.append( getattr(model, 'predict_proba'
                                     if hasattr(model, 'predict_proba')
                                 else 'predict')( x_test )
                            * ensemble_weights[i]
                        )
        preds = np.stack(preds).sum(axis = 0)
        predictions = preds
                
        print(
             'Total time used for testing: {:.1f} sec.' .format(
                             time.time() - test_begin))
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################



        
