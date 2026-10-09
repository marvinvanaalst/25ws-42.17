path = "/home/ss_students/mtp/modality_pipeline_wft_v3-main/data_modules/opus_CachingPatchDataset.py"
with open(path, "r") as f:
    code = f.read()

# Replace in _build_cache
code = code.replace(
    'v_min_source = source_tensor["v_min"].to(torch.float16)',
    'v_min_source = torch.as_tensor(source_tensor["v_min"], dtype=torch.float16, device=device)'
)
code = code.replace(
    'v_min_target = target_tensor["v_min"].to(torch.float16)',
    'v_min_target = torch.as_tensor(target_tensor["v_min"], dtype=torch.float16, device=device)'
)
code = code.replace(
    'v_max_source = source_tensor["v_max"].to(torch.float16)',
    'v_max_source = torch.as_tensor(source_tensor["v_max"], dtype=torch.float16, device=device)'
)
code = code.replace(
    'v_max_target = target_tensor["v_max"].to(torch.float16)',
    'v_max_target = torch.as_tensor(target_tensor["v_max"], dtype=torch.float16, device=device)'
)

# Replace in __getitem__
code = code.replace(
    '"v_min_source" : v_min_source.to(torch.float16)',
    '"v_min_source" : torch.as_tensor(v_min_source, dtype=torch.float16)'
)
code = code.replace(
    '"v_min_target": v_min_target.to(torch.float16)',
    '"v_min_target": torch.as_tensor(v_min_target, dtype=torch.float16)'
)
code = code.replace(
    '"v_max_source" : v_max_source.to(torch.float16)',
    '"v_max_source" : torch.as_tensor(v_max_source, dtype=torch.float16)'
)
code = code.replace(
    '"v_max_target": v_max_target.to(torch.float16)',
    '"v_max_target": torch.as_tensor(v_max_target, dtype=torch.float16)'
)

with open(path, "w") as f:
    f.write(code)

print("opus_CachingPatchDataset.py patched with torch.as_tensor for v_min/v_max!")
