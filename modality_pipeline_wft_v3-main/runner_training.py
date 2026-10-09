import torch            
from utils_custom.composite_loss import compile_loss_fn

def fetch_loss_local(experiment):
    return compile_loss_fn(experiment)

# One output Loss
def trainLoopFFLoss(model, loss_fn, optimizer, scheduler, scaler, batch):
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda"):
        output = model(batch['source'])
        loss = loss_fn(output.float(), batch['target'])
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    loss_value += loss.item() * batch['source'].shape[0]
    num_samples += batch['source'].shape[0]
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss

def valLoopFFLoss(model, loss_fn, batch):
    with torch.autocast(device_type="cuda"):  # Enable FP16-safe inference
        val_output = model(batch['source'])
        val_loss = loss_fn(val_output.float(), batch['target'].float())
    return val_loss.item()

def fetch_trainLoop_loss(experiment):
    if experiment['training_type'] == 'OneOutputLoss':
        return  fetch_loss_local(experiment), trainLoopFFLoss, valLoopFFLoss,


    



