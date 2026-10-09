import torch
import random

def load_training_state(path, model, optimizer, scheduler, scaler, device="cpu"):
    """_summary_

    Args:
        path (_type_): path to weights
        model (_type_): declared model
        optimizer (_type_): declared optimizer
        scheduler (_type_): declared scheduler
        scaler (_type_): declared scaler
        epoch (_type_): epoch
        device (str, optional): _description_. Defaults to "cpu".
    """

    ckpt = torch.load(path, map_location=device)   # weights_only=True default, fine
    
    assert "model" in ckpt, "Resuming not supported on old pipeline"
    
    for k, v in ckpt.items():
        print(k)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    if ckpt["scheduler"] and scheduler is not None: scheduler.load_state_dict(ckpt["scheduler"])
    torch.random.set_rng_state(ckpt["torch_rng"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    random.setstate(ckpt["python_rng"])
    epoch =  ckpt["epoch"]
    print(f"Resumed state at {epoch}")
    return epoch, ckpt['split']
    
def load_model_state(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)   # weights_only=True default, fine
    if ckpt.get("model", False):
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(model.state_dict())
    # return model, ckpt["step"], ckpt["epoch"]
    print(f"Loaded Model from {path}.")
