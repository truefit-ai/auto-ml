from typing import Any
from torch.utils.data import Dataset

class Metadata:
    def __init__(self, task_type, metric, min_size = None):
        self.task_type = task_type
        self.metric = metric
        self.min_size = min_size
        self.input_shape = None
        self.output_shape = None
        self.size_ = None
        
    def get_task_type(self):
        return self.task_type
    
    def get_final_metric(self):
        return self.metric
    
    def get_output_shape(self):
        return self.output_shape
    
    def get_tensor_shape(self):
        return self.input_shape
    
    def size(self):
        return self.size_

class AutoDataset(Dataset):
    def __init__(self, dataset, metadata):
        self.required_batch_size = None
        self.collate_fn = None
        self.metadata = metadata
        self.dataset = dataset
        self.metadata.input_shape = self.dataset[0][0].shape
        self.metadata.output_shape = self.dataset[0][1].shape
        self.metadata.size_ = len(self.dataset)


    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def get_metadata(self):
        return self.metadata
