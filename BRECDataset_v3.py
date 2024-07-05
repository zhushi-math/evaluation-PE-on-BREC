import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
import os
from scipy.special import comb


torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


def rrwp(adj, pe_len):
    adj = torch.from_numpy(adj).float()
    deg = adj.sum(dim=0)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float("inf")] = 0
    adj = deg_inv.reshape((-1, 1)) * adj
    pe_list = [torch.eye(adj.size(0))]
    out = adj
    pe_list.append(out)
    while len(pe_list) < pe_len:
        out = out @ adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=0)  # k x n x n
    return pe


def adj_powers(adj, pe_len):
    adj = torch.from_numpy(adj).float()
    deg = adj.sum(dim=0)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm_adj = deg_inv_sqrt.view((-1, 1)) * adj * deg_inv_sqrt.view((1, -1))
    pe_list = [torch.eye(adj.size(0))]
    out = norm_adj
    pe_list.append(out)
    while len(pe_list) < pe_len:
        out = out @ norm_adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=0)  # k x n x n
    return pe


def bern_poly(adj, pe_len):
    adj = torch.from_numpy(adj).float()
    deg = adj.sum(dim=0)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm_adj = deg_inv_sqrt.view((-1, 1)) * adj * deg_inv_sqrt.view((1, -1))
    eye = torch.eye(adj.size(0))
    adj1 = eye + norm_adj
    adj2 = eye - norm_adj
    K = pe_len - 2

    base1_list = [eye, adj1 / 2.] + [None] * (K - 1)
    base2_list = [eye, adj2 / 2.] + [None] * (K - 1)
    for k in range(2, K + 1):
        lidx, ridx = k // 2, k - k // 2
        base1_list[k] = base1_list[lidx] @ base1_list[ridx]
        base2_list[k] = base2_list[lidx] @ base2_list[ridx]

    bp_base_list = [base1_list[K - k] @ base2_list[k] for k in range(K + 1)]
    bp_coef_list = [comb(K, k) for k in range(K + 1)]
    basis = [bp_base_list[k] * bp_coef_list[k] for k in range(K + 1)]
    basis = [eye] + basis
    pe = torch.stack(basis, dim=0)  # K x n x n
    return pe


def mixed_bern_poly(adj, pe_len):
    K = pe_len - 2
    adj = torch.from_numpy(adj).float()
    deg = adj.sum(dim=0)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm_adj = deg_inv_sqrt.view((-1, 1)) * adj * deg_inv_sqrt.view((1, -1))
    eye = torch.eye(adj.size(0))
    adj1 = eye + norm_adj
    adj2 = eye - norm_adj

    base_list = [adj1 @ adj1, adj1 @ adj2, adj2 @ adj2]
    base_dict = {2: base_list}
    for k in range(4, K + 1, 2):
        a_idx = ((k // 2) + 1) // 2 * 2
        b_idx = k - a_idx
        base_list = [
            base_dict[a_idx][1] @ base_dict[b_idx][0],
            base_dict[a_idx][1] @ base_dict[b_idx][1],
            base_dict[a_idx][1] @ base_dict[b_idx][2],
        ]
        base_dict[k] = base_list

    polys = [adj1 / 2, adj2 / 2]
    for k in range(2, K + 1, 2):
        base_list = base_dict[k]
        base1 = base_dict[k][0] * ((2 ** -k) * comb(k, k // 2 - 1))
        base2 = base_dict[k][2] * ((2 ** -k) * comb(k, k // 2 + 1))
        polys += [base1, base2]

    pe = torch.stack(polys, dim=0)  # k x n x n
    return pe


def deco_bern_poly(adj, pe_len):
    K = pe_len - 2
    adj = torch.from_numpy(adj).float()
    deg = adj.sum(dim=0)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm_adj = deg_inv_sqrt.view((-1, 1)) * adj * deg_inv_sqrt.view((1, -1))
    eye = torch.eye(adj.size(0))
    adj1 = eye + norm_adj
    adj2 = eye - norm_adj

    base_list = [adj1 @ adj1, adj1 @ adj2, adj2 @ adj2]
    base_dict = {2: base_list}
    for k in range(4, K + 1, 2):
        a_idx = ((k // 2) + 1) // 2 * 2
        b_idx = k - a_idx
        base_list = [
            base_dict[a_idx][1] @ base_dict[b_idx][0],
            base_dict[a_idx][1] @ base_dict[b_idx][1],
            base_dict[a_idx][1] @ base_dict[b_idx][2],
        ]
        base_dict[k] = base_list

    polys = [adj1 / 2, eye, adj2 / 2]
    for k in range(2, K + 1, 2):
        base1 = base_dict[k][0] * ((2 ** -k) * comb(k, k // 2 - 1))
        base2 = base_dict[k][1] * ((2 ** -k) * comb(k, k // 2 + 0))
        base3 = base_dict[k][2] * ((2 ** -k) * comb(k, k // 2 + 1))
        polys += [base1, base2, base3]

    pe = torch.stack(polys, dim=0)  # (k*1.5) x n x n
    return pe


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        poly_method,
        pe_len,
        subset_index,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.poly_method = poly_method
        self.pe_len = pe_len
        self.subset_index = subset_index
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.my_process()
        # self._data = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3_numlabel=0.pt"]

    def my_process(self):

        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data = []
        for idx in self.subset_index:
            g = data_list[idx]
            g_networkx = nx.from_graph6_bytes(g)
            adj: np.ndarray = nx.to_numpy_array(g_networkx)
            if self.poly_method == 'rrwp':
                pe = rrwp(adj, self.pe_len)
            elif self.poly_method == 'adj_powers':
                pe = adj_powers(adj, self.pe_len)
            elif self.poly_method == 'bern_poly':
                pe = bern_poly(adj, self.pe_len)
            elif self.poly_method == 'mixed_bern_poly':
                pe = mixed_bern_poly(adj, self.pe_len)
            elif self.poly_method == 'deco_bern_poly':
                pe = deco_bern_poly(adj, self.pe_len)

            data.append(pe)

        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        data = self._data[idx]
        return data


def main():
    dataset = BRECDataset()
    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
