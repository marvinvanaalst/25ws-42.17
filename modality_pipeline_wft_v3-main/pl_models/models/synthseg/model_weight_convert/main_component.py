import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


import tensorflow as tf
from tensorflow.keras import layers, models

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

# def build_synthseg(input_shape=(None, None, None, 1), n_classes=33):
def build_synthseg(input_shape=(None, None, None, 1), n_classes=33):
    
    inputs = layers.Input(shape=input_shape, name='unet_input')

    # Encoder
    skip1, x = down_block(inputs, 24, 0)
    skip2, x = down_block(x, 48, 1)
    skip3, x = down_block(x, 96, 2)
    skip4, x = down_block(x, 192, 3)

    # Bottleneck
    x = conv_block(x, 384, 'unet_conv_downarm_4')
    x = layers.BatchNormalization(name='unet_bn_down_4')(x)

    # Decoder
    x = up_block(x, skip4, 192, 5)
    x = up_block(x, skip3, 96, 6)
    x = up_block(x, skip2, 48, 7)
    x = up_block(x, skip1, 24, 8)

    # Output
    x = layers.Conv3D(n_classes, 1, name='unet_likelihood')(x)
    outputs = layers.Softmax(axis=-1, name='unet_prediction')(x)

    return models.Model(inputs=inputs, outputs=outputs)

# Create the model
model = build_synthseg()


path = '/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/weights/synthseg_2.0.h5'
# path = '/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/weights/synthseg_parc_2.0.h5'
# path = '/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/weights/synthseg_qc_2.0.h5'
# path = '/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/weights/synthseg_robust_2.0.h5'
model.load_weights(path,  skip_mismatch=False)
model.summary()


import nibabel as nib
import numpy as np
import tensorflow as tf

sample_path = '/Drive4T/inam/MRMRData/T1/3T_t1/sub-01_T1w_defaced_registered.nii.gz'
sample_path ='/Drive4T/inam/MRMRData/T1/skull_stripped/Fin/3T_t1/native_res_152_sub-01_T1w_defaced_registered.nii_n4corrected.nii.gz'

# ---- Padding helpers ---- #
def pad_to_multiple(volume, multiple=16):
    """Pad a volume so each dim is divisible by `multiple`."""
    shape = volume.shape
    pad_width = []

    for dim in shape:
        total_pad = (multiple - dim % multiple) % multiple
        before = total_pad // 2
        after = total_pad - before
        pad_width.append((before, after))

    padded = np.pad(volume, pad_width, mode='reflect')
    return padded, pad_width

def unpad(volume, pad_width):
    """Remove padding to restore original shape."""
    slices = tuple(slice(p[0], -p[1] if p[1] > 0 else None) for p in pad_width)
    return volume[slices]

# ---- Model loading ---- #

# ---- Load + pad input ---- #
nifti = nib.load(sample_path)
input_vol = nifti.get_fdata().astype(np.float32)

# Normalize (SynthSeg expects normalized inputs)
input_vol = (input_vol - np.mean(input_vol)) / np.std(input_vol)

# Pad to multiple of 16
padded_vol, pad_info = pad_to_multiple(input_vol, multiple=16)

# Expand to batch & channel dims: (1, D, H, W, 1)
input_tensor = np.expand_dims(padded_vol, axis=(0, -1))

# ---- Predict ---- #
pred_probs = model.predict(input_tensor)  # Output: (1, D, H, W, num_classes)
pred_seg = np.argmax(pred_probs[0], axis=-1).astype(np.uint8)  # (D, H, W)

# ---- Crop back to original size ---- #
pred_seg = unpad(pred_seg, pad_info)

# ---- Save segmentation ---- #
output_nii = nib.Nifti1Image(pred_seg, affine=nifti.affine)
os.makedirs('meccano_output', exist_ok=True)
nib.save(output_nii, "meccano_output/synthseg_output.nii.gz")
nib.save(output_nii, "/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/synthseg_output.nii.gz")
try:
    nib.save(output_nii, "with_errir.nii.gz")
except Exception as e:
    print(f"Error saving NIfTI file: {e}")
nib.save(output_nii, "synthseg_output.nii.gz")
