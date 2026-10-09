import torch
import torch.nn as nn
import torch.nn.functional as F


def SynthSegCut(model, module_lim, layer_lim):
    cut_model = nn.Sequential()
    ia = 0
    for module_name, module in model.named_children():
        if ia < module_lim - 1:
            cut_model.add_module(module_name, module)
        elif ia == module_lim - 1:
            ib = 0
            new_module = nn.Sequential()
            for layer_name, layer in module.named_children():
                if ib < layer_lim:
                    new_module.add_module(layer_name, layer)
                ib += 1
            cut_model.add_module(module_name, new_module)

        ia += 1
    return cut_model


class LSynthSeg(nn.Module):
    # return the L1_loss between an image pair based on truncated SynthSeg Model
    # truncation of the SynthSeg Model is specified with the nblock and nlayer parameters
    def __init__(self, nblock, nlayer, loaded_model, device="cuda", type_loss='paper'):
        super(LSynthSeg, self).__init__()
        self.type_loss = type_loss
        truncated_SynthSeg = SynthSegCut(loaded_model, nblock, nlayer)
        if self.type_loss  == 'paper':
            self.truncated_SynthSeg = truncated_SynthSeg.eval().to(device)
        elif self.type_loss == 'custom':
            for param in truncated_SynthSeg.parameters():
                param.requires_grad = False
            self.truncated_SynthSeg = truncated_SynthSeg.eval().to(device)
    def forward(self, lr, hr):
        if self.type_loss == 'paper':
            with torch.no_grad():
                loss = F.l1_loss(self.truncated_SynthSeg(lr), self.truncated_SynthSeg(hr))
        elif self.type_loss == 'custom':
            loss = F.l1_loss(self.truncated_SynthSeg(lr), self.truncated_SynthSeg(hr))
        return loss
