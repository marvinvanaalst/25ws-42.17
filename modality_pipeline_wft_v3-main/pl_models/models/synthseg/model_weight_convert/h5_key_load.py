import h5py

path = '/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/weights/synthseg_2.0.h5'
f = h5py.File(path, "r")
for k, v in f.items():
    print(f"Key: {k}, Type: {type(v)}")
    # if isinstance(v):
    print(f"Shape: {v.shape}, Dtype: {v.dtype}")
    if isinstance(v, h5py.Group):
        print("This is a group containing:")
        for sub_k in v.keys():
            print(f"  Sub-key: {sub_k}")
print(list(f.keys()))
