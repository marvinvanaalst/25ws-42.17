import json

def update_cfg(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    if 'pretrain_augments' not in cfg:
        cfg['pretrain_augments'] = {}
    if 'block_augments' not in cfg['pretrain_augments']:
        cfg['pretrain_augments']['block_augments'] = {
            'apply': False,
            'config': {'gin_apply': False}
        }
    else:
        if 'config' not in cfg['pretrain_augments']['block_augments']:
            cfg['pretrain_augments']['block_augments']['config'] = {'gin_apply': False}
        else:
            cfg['pretrain_augments']['block_augments']['config']['gin_apply'] = False

    with open(path, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"Updated {path}")

update_cfg('/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/exp1_3d_vnet_l1_default.json')
update_cfg('/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/exp2_3d_vnet_l1_paper.json')

# Also fix line 123 of master_half_patch.py to be safe
code_path = '/home/ss_students/mtp/modality_pipeline_wft_v3-main/master_half_patch.py'
with open(code_path, 'r') as f:
    code = f.read()

code = code.replace("if block_config['gin_apply']:", "if block_config.get('gin_apply', False):")
with open(code_path, 'w') as f:
    code = f.write(code)

print("master_half_patch.py patched to safely check gin_apply!")
