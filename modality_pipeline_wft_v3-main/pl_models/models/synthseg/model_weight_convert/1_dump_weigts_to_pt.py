import tensorflow as tf
import torch
import numpy as np
from tensorflow.keras import layers, models

# --- Define TF Model ---
def conv_block(x, filters, name_prefix):
    x = layers.Conv3D(filters, 3, padding='same', name=f'{name_prefix}_0')(x)
    x = layers.ELU(alpha=1.0)(x)
    x = layers.Conv3D(filters, 3, padding='same', name=f'{name_prefix}_1')(x)
    x = layers.ELU(alpha=1.0)(x)
    return x

def down_block(x, filters, stage):
    conv = conv_block(x, filters, f'unet_conv_downarm_{stage}')
    pool = layers.MaxPooling3D(pool_size=2, name=f'unet_maxpool_{stage}')(conv)
    norm = layers.BatchNormalization(name=f'unet_bn_down_{stage}')(pool)
    return conv, norm

def up_block(x, skip, filters, stage):
    up = layers.UpSampling3D(size=2, name=f'unet_up_{stage}')(x)
    merge = layers.Concatenate(name=f'unet_merge_{stage}')([up, skip])
    conv = conv_block(merge, filters, f'unet_conv_uparm_{stage}')
    norm = layers.BatchNormalization(name=f'unet_bn_up_{stage}')(conv)
    return norm

def build_synthseg(input_shape=(None, None, None, 1), n_classes=33):
    inputs = layers.Input(shape=input_shape, name='unet_input')
    skip1, x = down_block(inputs, 24, 0)
    skip2, x = down_block(x, 48, 1)
    skip3, x = down_block(x, 96, 2)
    skip4, x = down_block(x, 192, 3)
    x = conv_block(x, 384, 'unet_conv_downarm_4')
    x = layers.BatchNormalization(name='unet_bn_down_4')(x)
    x = up_block(x, skip4, 192, 5)
    x = up_block(x, skip3, 96, 6)
    x = up_block(x, skip2, 48, 7)
    x = up_block(x, skip1, 24, 8)
    x = layers.Conv3D(n_classes, 1, name='unet_likelihood')(x)
    outputs = layers.Softmax(axis=-1, name='unet_prediction')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# --- Load model and weights ---
model = build_synthseg((128, 128, 128, 1), n_classes=33)
model.load_weights("/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/weights/synthseg_2.0.h5")  # <-- Change path to your file

# --- Extract weights ---
weights_dict = {}
for layer in model.layers:
    if layer.weights:
        weights_dict[layer.name] = [w.numpy() for w in layer.weights]

# --- Save to PyTorch format ---
torch.save(weights_dict, "synthseg_tf_weights.pt")
print("Saved weights to synthseg_tf_weights.pt")
