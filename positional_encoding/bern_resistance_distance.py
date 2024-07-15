import torch
import numpy as np
from scipy.special import comb


def compute_bernstein_polynomials_resistance_distance(adj, pe_power):
    K = pe_power
    adj = torch.from_numpy(adj).float()
    deg = adj.sum(dim=0)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm_adj = deg_inv_sqrt.view((-1, 1)) * adj * deg_inv_sqrt.view((1, -1))
    eye = torch.eye(adj.size(0))
    adj0 = eye + norm_adj
    adj1 = eye - norm_adj

    base1_list = [eye, adj0 / 2.] + [None] * (K - 1)
    base2_list = [eye, adj1 / 2.] + [None] * (K - 1)
    for k in range(2, K + 1):
        lidx, ridx = k // 2, k - k // 2
        base1_list[k] = base1_list[lidx] @ base1_list[ridx]
        base2_list[k] = base2_list[lidx] @ base2_list[ridx]

    bp_base_list = [base1_list[K - k] @ base2_list[k] for k in range(K + 1)]
    bp_coef_list = [comb(K, k) for k in range(K + 1)]
    basis = [bp_base_list[k] * bp_coef_list[k] for k in range(K + 1)]
    basis = [eye] + basis
    pe = torch.stack(basis, dim=0)  # (K+2) x n x n

    n = adj.shape[0]
    degree_matrix = torch.diag(adj.sum(dim=1))
    laplacian_matrix = degree_matrix - adj
    d_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree_matrix.diag()))
    laplacian_matrix = torch.matmul(torch.matmul(d_inv_sqrt, laplacian_matrix), d_inv_sqrt)

    # 计算拉普拉斯矩阵的伪逆
    laplacian_pseudo_inverse = torch.pinverse(laplacian_matrix)

    resistance_distances = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            resistance_distances[i, j] = (laplacian_pseudo_inverse[i, i] +
                                          laplacian_pseudo_inverse[j, j] -
                                          2 * laplacian_pseudo_inverse[i, j])
    
    resistance_distances = torch.unsqueeze(resistance_distances, 0)
    
    pe = torch.cat((pe, resistance_distances, 0))
    return pe