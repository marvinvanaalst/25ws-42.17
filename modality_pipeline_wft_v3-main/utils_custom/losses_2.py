import torch
import torch.nn.functional as F


class laplacian_loss:
    def __init__(self,device):
        self.kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=torch.float32).to(device)

    def laplacian_loss(self,pred, target):
        # Apply laplacian to both
        lap_pred = F.conv3d(pred, self.kernel, padding=1)
        lap_target = F.conv3d(target, self.kernel, padding=1)
        return F.l1_loss(lap_pred, lap_target)