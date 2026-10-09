import json

base_path = '/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/DS_1_VNET_L1.json'
with open(base_path, 'r') as f:
    cfg = json.load(f)

# Update for System 34 environment
cfg['experiment_name'] = 'exp1_3d_vnet_l1_default'
cfg['data']['training']['subjects_path'] = '/home/ss_students/mtp/10_Pat_t1-20260830T152711Z-1-001/10_Pat_t1'
cfg['device']['devices'] = ['cuda:0']

out_path1 = '/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/exp1_3d_vnet_l1_default.json'
with open(out_path1, 'w') as f:
    json.dump(cfg, f, indent=4)
print('Created:', out_path1)

# Create Exp 2 with paper parameters (Wolny / 3D UNet: lr=2e-4, weight_decay=1e-4)
cfg2 = json.loads(json.dumps(cfg))
cfg2['experiment_name'] = 'exp2_3d_vnet_l1_paper'
cfg2['training']['learning_rate'] = 0.0002
cfg2['training']['weight_decay'] = 0.0001
out_path2 = '/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/exp2_3d_vnet_l1_paper.json'
with open(out_path2, 'w') as f:
    json.dump(cfg2, f, indent=4)
print('Created:', out_path2)
