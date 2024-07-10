import torch


def diag_offdiag_maxpool(inputs):
    N = inputs.shape[-1]
    max_diag = torch.max(torch.diagonal(inputs, dim1=-2, dim2=-1), dim=2)[0]  # B, H
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * inputs)
    val = torch.abs(torch.add(max_val, min_val))
    min_mat = torch.mul(val, torch.eye(N, device=inputs.device)).view(1, 1, N, N)
    max_offdiag = torch.max(torch.max(inputs - min_mat, dim=3)[0], dim=2)[0]  # B, H
    return torch.cat((max_diag, max_offdiag), dim=1)  # output B, 2H


def diag_offdiag_meanpool(inputs, level='graph'):
    N = inputs.shape[-1]
    if level == 'graph':
        diag_sum = torch.sum(torch.diagonal(inputs, dim1=-2, dim2=-1), dim=2)  # B, H
        diag_avg = diag_sum / N
        offdiag_avg = (torch.sum(inputs, dim=[-1, -2]) - diag_sum) / (N * N - N)
        outputs = torch.cat((diag_avg, offdiag_avg), dim=1)  # B, 2H
    else:
        diag_val = torch.diagonal(inputs, dim1=-2, dim2=-1)  # B, H, N
        offdiag_val = (torch.sum(inputs, dim=-1) + torch.sum(inputs, dim=-2) - 2 * diag_val) / (2 * N - 2)
        outputs = torch.cat((diag_val, offdiag_val), dim=1)  # B, 2H, N
    return outputs
