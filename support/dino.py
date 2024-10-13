import torch

def get_features(model, x, features_type="x_norm_patchtokens", gpu_index=0):
    with torch.no_grad():
        output = model.forward_features(x.unsqueeze(0).cuda(gpu_index))
    return output[features_type].squeeze(0).cpu()