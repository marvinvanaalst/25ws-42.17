import torch
# from monai.networks.nets import UNETR
from pl_models.monai_edited.unetr import UNETR

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UNETR Configuration
# Note: Standard UNETR is large (~92M parameters). 
# To stay under 10M, you must reduce 'hidden_size' and 'mlp_dim'.
def fetch_UNETR(
    in_channels=1,        # e.g., for 3T or 7T MRI
    out_channels=2,       # Number of segmentation classes
    img_size=(96, 96, 96), # Input patch size
    feature_size=16,      # Width of the network layers
    hidden_size=768,      # Transformer embedding size (Reduce this for <10M)
    mlp_dim=3072,         # Transformer MLP dimension
    num_heads=12,         # Number of attention heads
    # pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
    interpolation=None
    ):

    # if interpolation == None:
    #     return UNETR(
    #         in_channels=in_channels,        # e.g., for 3T or 7T MRI
    #         out_channels=out_channels,       # Number of segmentation classes
    #         img_size=img_size, # Input patch size
    #         feature_size=feature_size,      # Width of the network layers
    #         hidden_size=hidden_size,      # Transformer embedding size (Reduce this for <10M)
    #         mlp_dim=mlp_dim,         # Transformer MLP dimension
    #         num_heads=num_heads,         # Number of attention heads
    #         norm_name=norm_name,
    #         # pos_embed=pos_embed,
    #         res_block=res_block,
    #         dropout_rate=dropout_rate
    #     )
    # else:
    return UNETR(
        in_channels=in_channels,        # e.g., for 3T or 7T MRI
        out_channels=out_channels,       # Number of segmentation classes
        img_size=img_size, # Input patch size
        feature_size=feature_size,      # Width of the network layers
        hidden_size=hidden_size,      # Transformer embedding size (Reduce this for <10M)
        mlp_dim=mlp_dim,         # Transformer MLP dimension
        num_heads=num_heads,         # Number of attention heads
        norm_name=norm_name,
        # pos_embed=pos_embed,
        res_block=res_block,
        dropout_rate=dropout_rate,
        # interpolation=interpolation
    )