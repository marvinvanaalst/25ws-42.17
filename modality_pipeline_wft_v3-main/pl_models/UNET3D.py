import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio
import pl_models.models.ESAU
class Simple3DUNet(pl.LightningModule):
    def __init__(self, params, lr=1e-3):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        in_channels = params['model']['input_channels']
        out_channels = params['model']['output_channels']
        
        
        self.lr = params['training']['learning_rate']
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(16, out_channels, kernel_size=1)
        )
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # input(batch)
        input_tensor = batch['source']
        # target_tensor = batch['target'][tio.DATA].float()
        target_tensor = batch['target']
        if batch['mask'] is not None:
            mask_tensor = batch['mask']
            input_tensor = input_tensor * mask_tensor
            target_tensor = target_tensor * mask_tensor
        
        output = self(input_tensor)
        loss = self.loss_fn(output, target_tensor)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor = batch['source']
        target_tensor = batch['target']
        if batch['mask'] is not None:
            mask_tensor = batch['mask']
            input_tensor = input_tensor * mask_tensor
            target_tensor = target_tensor * mask_tensor
        
        output = self(input_tensor)
        loss = self.loss_fn(output, target_tensor)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
