from pl_models.models.synthSeg import SynthSegModel
from pl_models.models.synthSeg_loss import LSynthSeg

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio

class ccx(pl.LightningModule):
    def __init__(
            self,
            params
    ):
        super(ccx, self).__init__()
        patj = '/storage/an_inam/MR2MR/patch_pipeline/pl_models/models/SynthSegWeights/synthseg_2.0.h5'
        SynthSeg_model = SynthSegModel()
        with open(patj, "rb") as f:
            print(f.read(10))  # Check the header
        ccxx = torch.load(patj, weights_only=False)
        print(ccxx, type(ccxx))
        SynthSeg_model.load_state_dict((ccxx), weight_only=False)
        SynthSeg_model.eval()

        print(SynthSeg_model)
        input("Hcxm")

    def forward(self, x):
        return self.net_G(x)

    def configure_optimizers(self):
        lr = self.lr
        l2 = self.l2
        opt_g = torch.optim.RMSprop(self.net_G.parameters(), lr=lr, weight_decay=l2)
        opt_d = torch.optim.RMSprop(self.net_D.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def on_after_backward(self):
        clip_value = self.weight_clipping
        # Clip discriminator weights after each backward pass
        for p in self.net_D.parameters():
            p.data.clamp_(-clip_value, clip_value)

    def training_step(self, batch, batch_idx):
        
        g_opt, d_opt = self.optimizers()

        img_3t = batch['t1_3T'][tio.DATA]
        img_7t = batch['t1_7T'][tio.DATA]
        mask = batch['mask'][tio.DATA]

        self.toggle_optimizer(g_opt)
        img_s7t = self.forward(img_3t) * mask

        # loss
        MAE_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())

        if self.loss_type == 'MAE':
            content_loss = MAE_loss
        elif self.loss_type == 'SynthSeg3':
            SynthSeg_loss = self.criterion_SynthSeg_AfterBlock3(img_s7t.squeeze(), img_7t.squeeze())
            content_loss = self.MAE_alpha * MAE_loss + self.SynthSeg_alpha * SynthSeg_loss

        adv_loss = -torch.mean(self.net_D(img_s7t))
        g_loss = content_loss + adv_loss
        self.manual_backward(g_loss)
        g_opt.step()
        g_opt.zero_grad()
        self.untoggle_optimizer(g_opt)

        self.toggle_optimizer(d_opt)
        d_loss = -torch.mean(self.net_D(img_7t)) + torch.mean(self.net_D(img_s7t.detach()))
        self.manual_backward(d_loss)
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

        bn = self.bn
        self.log('train_MAE', MAE_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('train_generator_Loss', g_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('train_discriminator_Loss', d_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('train_adv_loss', adv_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)

        if self.loss_type == 'SynthSeg3':
            self.log('train_SynthSegLoss', SynthSeg_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=bn, sync_dist=True)
        return g_loss

    def validation_step(self, batch, batch_idx):
        img_3t = batch['t1_3T'][tio.DATA]
        img_7t = batch['t1_7T'][tio.DATA]
        mask = batch['mask'][tio.DATA]
        img_s7t = self.forward(img_3t) * mask

        # loss
        MAE_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())

        if self.loss_type == 'MAE':
            content_loss = MAE_loss
        elif self.loss_type == 'SynthSeg3':
            SynthSeg_loss = self.criterion_SynthSeg_AfterBlock3(img_s7t.squeeze(), img_7t.squeeze())
            content_loss = self.MAE_alpha * MAE_loss + self.SynthSeg_alpha * SynthSeg_loss

        adv_loss = -torch.mean(self.net_D(img_s7t))
        g_loss = content_loss + adv_loss
        d_loss = -torch.mean(self.net_D(img_7t)) + torch.mean(self.net_D(img_s7t.detach()))

        bn = self.bn
        self.log('val_MAE', MAE_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('val_generator_Loss', g_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('val_discriminator_Loss', d_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('val_adv_loss', adv_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)

        if self.loss_type == 'SynthSeg3':
            self.log('val_SynthSegLoss', SynthSeg_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=bn, sync_dist=True)
        return g_loss

if __name__ == "__main__":
    llxx = ccx(None)
    