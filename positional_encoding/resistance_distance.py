import torch

def resistance_distance(adj, _):
    adj = torch.from_numpy(adj).float()
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
    
    return resistance_distances


