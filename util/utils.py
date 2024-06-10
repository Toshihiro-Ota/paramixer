import torch

def reg_weight2(
        model,
        _lambda,
        ):
    seq = model.blocks
    temp = torch.sum(seq[0].mlp_tokens.fc2.weight **2) + torch.sum(seq[0].mlp_channels.fc2.weight **2)
    for block in seq[1:]:
        temp += torch.sum(block.mlp_tokens.fc2.weight **2) + torch.sum(block.mlp_channels.fc2.weight **2)
    return _lambda * temp
