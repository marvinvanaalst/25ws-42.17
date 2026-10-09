#!/usr/bin/env python3
"""
patch_master_for_transunet.py
Applies two backward-compatible patches to master_half_slice_re.py:
  1. build_optimizer: adds SGD (with Nesterov) and plain Adam support.
  2. get_lr_scheduler: adds polynomial decay support via config key
     training.lr_schedule.type == 'poly_decay'.

Run once on the server:
    cd /home/ss_students/mtp/modality_pipeline_wft_v3-main
    python patch_master_for_transunet.py
"""

import re, sys, shutil, os
from pathlib import Path

TARGET = Path("/home/ss_students/mtp/modality_pipeline_wft_v3-main/master_half_slice_re.py")
BACKUP = TARGET.with_suffix(".py.bak_transunet")

def patch():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        sys.exit(1)

    src = TARGET.read_text()

    # -- Guard: don't double-patch --
    if "TRANSUNET_PATCH_APPLIED" in src:
        print("Already patched — nothing to do.")
        return

    # ==================================================================
    # Patch 1: build_optimizer — add SGD + Adam
    # ==================================================================
    OLD_OPT = '''\
def build_optimizer(model):
    name = str(experiment['training'].get('optimizer', 'adamw')).lower()
    kwargs = dict(lr=experiment['training']['learning_rate'],
                  weight_decay=experiment['training']['weight_decay'])
    # if name == 'adam':
    #     return torch.optim.Adam(model.parameters(), **kwargs)
    return torch.optim.AdamW(model.parameters(), **kwargs)'''

    NEW_OPT = '''\
def build_optimizer(model):  # TRANSUNET_PATCH_APPLIED
    name = str(experiment['training'].get('optimizer', 'adamw')).lower()
    lr  = experiment['training']['learning_rate']
    wd  = experiment['training']['weight_decay']
    if name == 'sgd':
        momentum = float(experiment['training'].get('momentum', 0.9))
        print(f"[Optimizer] SGD lr={lr} momentum={momentum} wd={wd} nesterov=True")
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                               momentum=momentum, nesterov=True)
    if name == 'adam':
        print(f"[Optimizer] Adam lr={lr} wd={wd}")
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # default: AdamW
    print(f"[Optimizer] AdamW lr={lr} wd={wd}")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)'''

    if OLD_OPT not in src:
        print("WARNING: build_optimizer block not found exactly — trying regex fallback")
        # Regex fallback: replace from 'def build_optimizer' to first blank line after return
        src_new = re.sub(
            r'(def build_optimizer\(model\):.*?return torch\.optim\.AdamW\(model\.parameters\(\),\s*\*\*kwargs\))',
            NEW_OPT,
            src,
            flags=re.DOTALL,
        )
        if src_new == src:
            print("ERROR: Could not patch build_optimizer. Aborting.")
            sys.exit(1)
        src = src_new
        print("  Patched build_optimizer (regex).")
    else:
        src = src.replace(OLD_OPT, NEW_OPT, 1)
        print("  Patched build_optimizer (exact match).")

    # ==================================================================
    # Patch 2: get_lr_scheduler — add poly_decay support
    # ==================================================================
    # We inject the poly_decay branch INSIDE get_lr_scheduler, just before
    # the final `return LambdaLR(optimizer, lr_lambda)`.
    # The existing lr_lambda handles linear_warmup + optional step decay.
    # For poly_decay we return a separate LambdaLR immediately.
    INJECT_BEFORE = "    return LambdaLR(optimizer, lr_lambda)"

    POLY_BLOCK = '''\
    # ---- Polynomial decay (for SGD, matching TransUNet official repo) ----
    sched_type = experiment['training'].get('lr_schedule', {}).get('type', 'linear_warmup')
    if sched_type == 'poly_decay':
        total_steps = max(1, int(experiment['training']['epochs']) * max(1, int(steps_per_epoch)))
        power = float(experiment['training'].get('lr_schedule', {}).get('poly_power', 0.9))
        print(f"[Scheduler] poly_decay power={power} total_steps={total_steps}")
        def _poly_lr(step):
            return max(0.0, (1.0 - step / total_steps) ** power)
        return LambdaLR(optimizer, _poly_lr)

    '''

    if INJECT_BEFORE not in src:
        print("WARNING: get_lr_scheduler return not found exactly. Poly-decay patch skipped.")
        print("  You can still run exp9a — it will use the linear_warmup scheduler instead.")
    else:
        src = src.replace(INJECT_BEFORE, POLY_BLOCK + INJECT_BEFORE, 1)
        print("  Patched get_lr_scheduler (poly_decay added).")

    # ==================================================================
    # Write out
    # ==================================================================
    shutil.copy(TARGET, BACKUP)
    print(f"  Backup saved: {BACKUP}")
    TARGET.write_text(src)
    print(f"  Written: {TARGET}")
    print("\nPatch complete. Run a quick sanity check:")
    print("  python -c \"import master_half_slice_re\" 2>&1 | head -5")

if __name__ == "__main__":
    patch()
