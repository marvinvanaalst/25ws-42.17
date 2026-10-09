import os
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
import json
import torchio as tio
import random
from data_modules.opus_CachingSliceDataset import CACHE_MODE_ALIASES, resolve_cache_mode, cache_slice_dataloader

class CachingSliceDataset25D(Dataset):
    def __init__(self, config, sample_list, device, data_grouping='slices', type_load='all',
                 affine=None, cache_mode=None, num_slices=5):
        if not (device.startswith("cuda") or device.startswith("cpu")):
            raise ValueError(f"Invalid device: {device}")

        self.config = config
        self.device = device
        self.cache_mode = cache_mode or resolve_cache_mode(config)
        self.map_location = device if self.cache_mode == 'same_gpu' else 'cpu'
        self.complete_cache = self.cache_mode != 'none'
        self.num_slices = num_slices
        self.half_window = num_slices // 2

        self.cache = []
        self.slice_groups = []

        slice_var = f"{self.config['slicify']['resize_to']}x{self.config['slicify']['pad_to']}"
        data_dir = f"{self.config['data']['training']['subjects_path']}/{data_grouping}/"
        source_dir = os.path.join(data_dir, slice_var, f"{self.config['data']['direction']['source']}/{type_load}")
        target_dir = os.path.join(data_dir, slice_var, f"{self.config['data']['direction']['target']}/{type_load}")
        mask_dir   = os.path.join(data_dir, slice_var, f"{self.config['data']['direction']['mask_dir_name']}/{type_load}")

        self._index_files(source_dir, target_dir, mask_dir, sample_list)
        if self.complete_cache:
            self._build_cache()

    def _index_files(self, source_dir, target_dir, mask_dir, sample_list):
        for sample_name in sample_list:
            source_path = os.path.join(source_dir, sample_name)
            target_path = os.path.join(target_dir, sample_name)
            mask_path   = os.path.join(mask_dir,   sample_name)

            if not os.path.exists(source_path):
                continue
            pt_files = sorted([f for f in os.listdir(source_path) if f.endswith('.pt')])
            num_files = len(pt_files)

            for i, pt_file in enumerate(pt_files):
                target_file = os.path.join(target_path, pt_file)
                mask_file   = os.path.join(mask_path,   pt_file)

                if os.path.isfile(target_file) and os.path.isfile(mask_file):
                    src_window = []
                    for offset in range(-self.half_window, self.half_window + 1):
                        clamped_idx = max(0, min(num_files - 1, i + offset))
                        src_window.append(os.path.join(source_path, pt_files[clamped_idx]))
                    self.slice_groups.append((src_window, target_file, mask_file))

    def _load(self, idx):
        src_files, target_file, mask_file = self.slice_groups[idx]

        src_tensors = []
        first_source = None
        for s_file in src_files:
            s_data = torch.load(s_file, map_location=self.map_location)
            if first_source is None:
                first_source = s_data
            t = s_data['data'] if 'data' in s_data else s_data['map']
            if t.dim() == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            src_tensors.append(t)

        stacked_source = torch.stack(src_tensors, dim=0)

        target_tensor = torch.load(target_file, map_location=self.map_location)
        mask_tensor   = torch.load(mask_file,   map_location=self.map_location)

        out_dict = {
            'source': stacked_source,
            'target': target_tensor['data'],
            'mask':   mask_tensor['data'],
        }
        if 'affine' in first_source:
            out_dict['affine'] = first_source['affine']
            out_dict['original_size'] = first_source['original_size']
            out_dict['padding'] = first_source['padding']
            out_dict['path_scan'] = first_source['path_scan']
            out_dict['path_mask'] = first_source['path_mask']
            out_dict['target_affine'] = target_tensor['affine']

        return out_dict

    def _build_cache(self):
        loading_loop = tqdm(range(len(self.slice_groups)),
                            desc=f"[{self.cache_mode.upper()}-2.5D] Caching", unit="slice")
        for idx in loading_loop:
            self.cache.append(self._load(idx))

    def __len__(self):
        return len(self.cache) if self.complete_cache else len(self.slice_groups)

    def __getitem__(self, idx):
        return self.cache[idx] if self.complete_cache else self._load(idx)


class cache_slice_25d_dataloader(cache_slice_dataloader):
    def _make_dataset(self, sample_list, type_load, affine=None):
        return CachingSliceDataset25D(self.params, sample_list, self.device,
                                      type_load=type_load, affine=affine,
                                      cache_mode=self.cache_mode, num_slices=5)
