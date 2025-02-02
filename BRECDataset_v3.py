import os

import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

from positional_encoding import pe_computer_dict

torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        poly_method,
        pe_power,
        subset_index,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.poly_method = poly_method
        self.pe_power = pe_power
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
            pe = pe_computer_dict[self.poly_method](adj, self.pe_power)
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
