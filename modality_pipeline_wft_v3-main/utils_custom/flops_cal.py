def count_true_flops(model, input_tensor):
    from torchtnt.utils.flops import FlopTensorDispatchMode
    with FlopTensorDispatchMode(model) as mode:
        _ = model(input_tensor)
    # return mode.get_total_flops(), mode.get_module_flops()
    print(f"Total FLOPs: {mode.get_total_flops()/1e6:.3f} MFLOPs")

# # Example
# total, per_module = count_true_flops(model, x)
# print(f"Total FLOPs: {total/1e6:.3f} MFLOPs")
