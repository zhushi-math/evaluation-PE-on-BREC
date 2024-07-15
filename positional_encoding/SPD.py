import torch


def SPD_floyd_warshall(adj, _):
    n = len(adj)
    distance = [[adj[i][j] for j in range(n)] for i in range(n)]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

    return torch.tensor(distance, dtype=torch.float32).unsqueeze(0)

