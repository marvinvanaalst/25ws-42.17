import os
import heapq
import torch
import numpy as np
from torchvision.utils import save_image

import random
class LightningStyleCheckpoint:
    def __init__(self, 
                 save_dir,
                 monitor="val_ssim", 
                 mode="min",
                 top_k=0,
                 save_every_n_epochs=None,
                 save_weights_only=False,
                 filename_template="epoch={epoch:02d}_{monitor}={score:.4f}.pt",
                 save_every_template="epoch={epoch:02d}.pt"):
        """
        save_every_n_epochs: if set, saves every N epochs in addition to Top-K
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n\Weight Save Config:")
        print(f"Monitoring {monitor}")
        print(f"Mode {mode}")
        print(f"Heaping {top_k}")
        print(f"Saving Every {save_every_n_epochs}\n\n")
        
        
        
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k
        self.save_every_n_epochs = save_every_n_epochs
        self.save_weights_only = save_weights_only
        self.filename_template = filename_template
        self.save_every_template = save_every_template
        self.saved = []  # (score, path) heap
        self.is_min_mode = mode == "min"

    def save(self, model, optimizer, scheduler, scaler, step, epoch, split, logs: dict):
        ckpt = {
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scaler":       scaler.state_dict(),
            "scheduler":    scheduler.state_dict() if scheduler else None,
            "step":         step,
            "epoch":        epoch,
            "split": split,
            "torch_rng":    torch.random.get_rng_state(),
            "cuda_rng":     torch.cuda.get_rng_state_all(),
            "python_rng":   random.getstate(),
        }
        
        should_save_regular =  (
            self.save_every_n_epochs is not None and epoch % self.save_every_n_epochs == 0)
        filename = self.save_every_template.format(epoch=epoch)
        
        if should_save_regular:
            # print(f"[Checkpoint] Periodic Save: Epoch {epoch}")
            save_path = f"{self.save_dir}/saveEvery_epoch_{filename}"
            print("Checkpoint created at", f"{self.save_dir}/saveEvery_epoch_{filename}")
            torch.save(ckpt, save_path)
        
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        #     # print(f"[Checkpoint] Warning: {self.monitor} not in logs. Skipping.")
        #     # print(f"Available log keys: {list(logs.keys())}")
        #     return

        # Format file name
        # filename = self.filename_template.format(epoch=epoch, monitor=self.monitor, score=current_score)

        # Now Top-K logic
        save_path = f"{self.save_dir}/topk{self.monitor}_{filename}"
        # if np.isnan(current_score):
        #     return
        heap_score = current_score if not self.is_min_mode else -1*current_score
        heapq.heappush(self.saved, (heap_score, save_path))
        
        if self.top_k > 0:
            if len(self.saved) > self.top_k:
                # Remove worst checkpoint
                worst = heapq.heappop(self.saved)
                if os.path.exists(worst[1]):
                    os.remove(worst[1])
                    # print(f"[Checkpoint] Removed (top_k): {worst[1]}")
                # else:
                #     print(f"[Checkpoint] File not found, could not remove: {worst[1]}")

            top_k_boolean = False
            if any(path == save_path for _, path in self.saved):
                top_k_boolean = True
                # if self.save_weights_only:
                #     torch.save(ckpt['model'], save_path)
                # else:
                torch.save(ckpt, save_path)
                # print(f"[Checkpoint] Saved (top_k): {save_path}")
            else:
                # print(f"[Checkpoint] Skipped saving {save_path} — not in top-{self.top_k}")
                pass
            
            if top_k_boolean:
                return
            
        # if should_save_regular:
        #     # print(f"[Checkpoint] Periodic Save: Epoch {epoch}")
        #     save_path = f"{self.save_dir}/saveEvery_epoch_{filename}"
            
        #     if self.save_weights_only:
        #         torch.save(model.state_dict(), save_path)
        #     else:
        #         torch.save(model, save_path)
        #     # print(f"[Checkpoint] Saved (periodic): {save_path}")