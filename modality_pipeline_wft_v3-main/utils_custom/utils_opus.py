import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio

import sys
sys.path.append('/storage/an_inam/MR2MR/patch_pipeline')
print(sys.path)
import pl_models.models.ESAU_net
from torch.utils.data import DataLoader

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import torchio as tio
import nibabel as nib
import importlib.util
import os
import inspect
from tqdm import tqdm
def pruge_extra_channels(tensor1, tensor2):
    """
    Prune extra channels from the tensors if they have more than one channel.
    """
    if tensor1.shape[0] != 1:
        print(f"Expected tensor1 to have batch size 1, but got {tensor1.shape[0]} channels.")
    if tensor2.shape[0] != 1:
        print(f"Expected tensor2 to have batch size 1, but got {tensor2.shape[0]} channels.")
    while len(tensor1.shape) != len(tensor2.shape):
        if len(tensor1.shape) > len(tensor2.shape):
            if tensor1.shape[1] == 1:
                tensor1 = tensor1.squeeze(1)
            else:
                tensor1 = tensor1.squeeze(0)
        elif len(tensor2.shape) > len(tensor1.shape):
            if tensor2.shape[1] == 1:
                tensor2 = tensor2.squeeze(1)
            else:
                tensor2 = tensor2.squeeze(0)
    return tensor1, tensor2

def dyanamic_import_model(model_path, model_class):
    """
    Dynamically import a model from a given path.
    """
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, model_class)
    return model_class

def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    valid_params = sig.parameters
    return {k: v for k, v in kwargs.items() if k in valid_params}

class model_fetcher():
    def __init__(self, params):
        self.params = params
        # print(params['model_info'])

        # Intializing model from model path given as argument in params
        model_path = params['model_info']['path'] 
        model_class = params["model_info"]["model_class"]
        
        self.model_params = filter_kwargs(model_class.__init__, params['model'])
        self.model_params= params['model']
        self.model = dyanamic_import_model(model_path, model_class)
    
    def fetch_model(self):
        
        return self.model, self.model_params
