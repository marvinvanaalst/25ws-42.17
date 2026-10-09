import json

def update_cfg(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    cfg['sample_volume_during_training'] = {
        'do': False,
        'every': 10
    }
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"Added sample_volume_during_training to {path}")

update_cfg('/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/exp1_3d_vnet_l1_default.json')
update_cfg('/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/exp2_3d_vnet_l1_paper.json')

code_path = '/home/ss_students/mtp/modality_pipeline_wft_v3-main/master_half_patch.py'
with open(code_path, 'r') as f:
    code = f.read()

target = 'print({type(experiment["sample_volume_during_training"]["every"])}, type(experiment["sample_volume_during_training"]["every"]))'
if target in code:
    code = code.replace(target, '# ' + target)
    with open(code_path, 'w') as f:
        f.write(code)
    print("Commented out debug print in master_half_patch.py")
else:
    print("Debug print not found or already commented")
