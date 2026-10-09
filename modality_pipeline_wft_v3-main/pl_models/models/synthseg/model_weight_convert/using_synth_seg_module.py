import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio
from .VNet import VNet
from pl_models.models.synthseg.SynthSeg_parts import SynthSegModel
from SynthSeg_loss import LSynthSeg

path = '/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/weights/synthseg_2.0.h5'

SynthSeg_model = SynthSegModel()
SynthSeg_model.load_state_dict(torch.load(path))
SynthSeg_model.eval()