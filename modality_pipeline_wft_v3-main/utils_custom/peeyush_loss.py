from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
import torch
import torch.nn as nn

def define_pretrained(model_name):
    if model_name == 'vgg19':
        pretrained = torchvision.models.vgg19(weights=torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet152':
        pretrained = torchvision.models.resnet152(weights=torchvision.models.resnet.ResNet152_Weights)
    return pretrained

class PerceptionLoss2D(nn.Module):
    def __init__(
        self,
        feature_extractor,
        loss_fn=nn.L1Loss(),
        channel_dim: int = 3,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        separate_channel: bool = True,
        base_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.channel_dim = channel_dim
        
        if normalize_mean is not None:
            c = len(normalize_mean)
            self.normalize_mean = torch.Tensor(normalize_mean).view(1, c, 1, 1)
            self.normalize_std = torch.Tensor(normalize_std).view(1, c, 1, 1)
        else:
            self.normalize_mean = normalize_mean
            self.normalize_std = normalize_std
        
        self.separate_channel = separate_channel
        self.base_weight = base_weight
    
    def forward(self, out, target, return_record=False):
        if isinstance(out, dict):
            out = out['level_0']
        if isinstance(target, dict):
            target = target['level_0']
            
        if self.normalize_mean is not None:
            device = out.device
            self.normalize_mean = self.normalize_mean.to(device)
            self.normalize_std = self.normalize_std.to(device)
            out = (out - self.normalize_mean) / (self.normalize_std + 1e-5)
            target = (target - self.normalize_mean) / (self.normalize_std + 1e-5)
        
        # Handle channel dimension for 2D images
        b, c, h, w = out.shape
        if c != self.channel_dim or self.separate_channel:
            # Convert grayscale to RGB by repeating channels
            if c == 1:
                out = out.repeat(1, self.channel_dim, 1, 1)
                target = target.repeat(1, self.channel_dim, 1, 1)
        
        # Extract features
        o_features = self.feature_extractor(out)
        t_features = self.feature_extractor(target)
        
        # Calculate perceptual loss
        loss = 0
        for key in o_features.keys():
            loss += self.loss_fn(o_features[key], t_features[key]) / self.base_weight
        
        loss_record = loss.item()
        
        if return_record:
            return loss, loss_record
        else:
            return loss
        
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
        # Create VGG19 feature extractor
        model_name = 'vgg19'
        return_nodes = ['features.35']  # Single layer for reduced complexity
        
        pretrained = define_pretrained(model_name).eval()
        feature_extractor = create_feature_extractor(pretrained, return_nodes)
        feature_extractor.to(device)
        
        self.perceptual_loss = PerceptionLoss2D(
            feature_extractor=feature_extractor,
            loss_fn=nn.L1Loss(),
            channel_dim=3,
            separate_channel=True,
            base_weight=1.0  # Single layer so base_weight = 1
        )
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = self.l1_weight * l1 + self.perceptual_weight * perceptual
        return total_loss
    