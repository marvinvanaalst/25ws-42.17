import torch

# Load the saved .pt file from TensorFlow model
keras_weights = torch.load("/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/synthseg_tf_weights.pt", map_location="cpu", weights_only=False)

# Placeholder for converted weights
converted_weights = {}

# Helper function to transpose Conv3D kernel
def transpose_conv3d_weight(weight_np):
    # TensorFlow: (D, H, W, in_c, out_c) → PyTorch: (out_c, in_c, D, H, W)
    return torch.tensor(weight_np).permute(4, 3, 0, 1, 2)

# --------------------
# Mapping function
# --------------------
def map_conv(layer_tf_name, torch_module_name):
    w, b = keras_weights[layer_tf_name]
    converted_weights[f"{torch_module_name}.weight"] = transpose_conv3d_weight(w)
    converted_weights[f"{torch_module_name}.bias"] = torch.tensor(b)

def map_bn(layer_tf_name, torch_module_name):
    gamma, beta, mean, var = keras_weights[layer_tf_name]
    converted_weights[f"{torch_module_name}.weight"] = torch.tensor(gamma)  # scale
    converted_weights[f"{torch_module_name}.bias"]   = torch.tensor(beta)
    converted_weights[f"{torch_module_name}.running_mean"] = torch.tensor(mean)
    converted_weights[f"{torch_module_name}.running_var"]  = torch.tensor(var)

# --------------------
# InputTransitionSS
# --------------------
map_conv("unet_conv_downarm_0_0", "in_tr.conv1")
map_conv("unet_conv_downarm_0_1", "in_tr.conv2")

# --------------------
# Down blocks
# --------------------
map_bn("unet_bn_down_0", "down_tr48.bn1")
map_conv("unet_conv_downarm_1_0", "down_tr48.conv1")
map_conv("unet_conv_downarm_1_1", "down_tr48.conv2")

map_bn("unet_bn_down_1", "down_tr96.bn1")
map_conv("unet_conv_downarm_2_0", "down_tr96.conv1")
map_conv("unet_conv_downarm_2_1", "down_tr96.conv2")

map_bn("unet_bn_down_2", "down_tr192.bn1")
map_conv("unet_conv_downarm_3_0", "down_tr192.conv1")
map_conv("unet_conv_downarm_3_1", "down_tr192.conv2")

map_bn("unet_bn_down_3", "down_tr384.bn1")
map_conv("unet_conv_downarm_4_0", "down_tr384.conv1")
map_conv("unet_conv_downarm_4_1", "down_tr384.conv2")

map_bn("unet_bn_down_4", "bn_bottleneck")

# --------------------
# Up blocks
# --------------------
map_conv("unet_conv_uparm_5_0", "up_tr192.conv1")
map_conv("unet_conv_uparm_5_1", "up_tr192.conv2")
map_bn("unet_bn_up_5", "up_tr192.bn1")

map_conv("unet_conv_uparm_6_0", "up_tr96.conv1")
map_conv("unet_conv_uparm_6_1", "up_tr96.conv2")
map_bn("unet_bn_up_6", "up_tr96.bn1")

map_conv("unet_conv_uparm_7_0", "up_tr48.conv1")
map_conv("unet_conv_uparm_7_1", "up_tr48.conv2")
map_bn("unet_bn_up_7", "up_tr48.bn1")

# --------------------
# Final output conv
# --------------------
map_conv("unet_conv_uparm_8_0", "conv1")
map_conv("unet_conv_uparm_8_1", "conv2")
map_bn("unet_bn_up_8", "bn1")

w, b = keras_weights["unet_likelihood"]
converted_weights["conv_likelihood.weight"] = transpose_conv3d_weight(w)
converted_weights["conv_likelihood.bias"]   = torch.tensor(b)

# --------------------
# Save converted weights
# --------------------
torch.save(converted_weights, "synthseg_pytorch_weights.pt")
print("✅ PyTorch weights saved to synthseg_pytorch_weights.pt")
