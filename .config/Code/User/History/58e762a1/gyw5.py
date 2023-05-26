from typing import Dict, Tuple, Any, List
from pathlib import Path
import os

import wandb
import networkx as nx
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import omegaconf
import hydra
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import torch_geometric.transforms as T
import torch_geometric as tg

from tqdm import tqdm

from src.data.SpreadModel import KarateSpreadModel

# class KarateClubSimpleDataset(InMemoryDataset):
class KarateClubSimpleDatasetTransductive():
    """
    Dataset for the Karate Club graph.
    NOTE: Transductive learning is used, i.e. the model is trained on the entire graph.
    Same number of nodes and connectivity in each graph.
    Transductive refers to the fact that all spread realizations are observed during training.
    """


    def __init__(self, config: DictConfig, transform=None, pre_transform=None):
        self.config = config
        self.data: List[Data] = []
        # super().__init__(None, transform, pre_transform)
        
        # print(self.processed_paths)

        self.num_graphs = None
        self.num_nodes = None
        self.num_features = None
        self.num_classes = None
        self.data: List[Data] = []

        
    def process(self):
        save_dir = Path(self.config.data_save_dir)
        # self.processed_dir = save_dir / config.dataset.processed_dir
        
        if self.config.dataset.plot:
            ncols = 3
            nrows = self.config.dataset.num_graphs // ncols + (self.config.dataset.num_graphs % ncols > 0)
            fig, axs = plt.subplots(ncols, nrows, figsize=(5, 3))
        
        if self.config.dataset.load_data:
            for file in os.listdir(save_dir):
                if file.startswith('karate_spread_simple'):
                    self.data.append(torch.load(save_dir / file))
                    
            if len(self.data) == 0:
                print('No data found')
                return
        
        else:

            # Train/validation/test split
            transform_split = T.RandomNodeSplit(split='random', num_splits=1,
                        num_val=0.2, num_test=0.2, num_train_per_class=10)
            transform_degree_features = T.OneHotDegree(max_degree=20)
            
            transform = T.Compose([
                transform_split,
                transform_degree_features
            ])
            
            model = KarateSpreadModel(self.config, p_informed_activation=self.config.dataset.p_informed_activation,
                                    p_spontaneous_activation=self.config.dataset.p_spontaneous_activation)

            # source_pool = model.rng.choice(list(model.G.nodes), 
            #     size=self.config.dataset.num_initial_informed*self.config.dataset.num_graphs, replace=False)
            source_pool = list(model.G.nodes)

            for graph_idx in range(self.config.dataset.num_graphs):

                # Sample initial informed nodes
                sources = ['3']
                # sources = source_pool[graph_idx*self.config.dataset.num_initial_informed:(graph_idx+1)*self.config.dataset.num_initial_informed]
                print(sources)
                informed_activations, spontaneous_activations, used_links = model.spread(
                    infected=sources)

                G: nx.Graph = model.G
                
                # Generate data matrix for torch_geometric
                y: torch.Tensor = torch.zeros((len(G.nodes), 1))
                for node in G.nodes:
                    # Set node features
                    y[int(node)] = int(node in informed_activations or node in spontaneous_activations)
                    
                for edge in G.edges:
                    # Set edge features
                    G.edges[edge]['used_edge'] = int(edge in used_links)
                    G.edges[edge]['transmission_rate'] = model.transmission_rates[edge]
                
                data: Data = tg.utils.from_networkx(G)
                data.validate(raise_on_error=True)
                data.y = y.squeeze().long()
                data.x = torch.zeros_like(data.y, dtype=torch.float)
                data.x[int(sources[0])] = 1.0
                data = transform(data)

                self.data.append(data)

                # save_path = hydra.utils.to_absolute_path(config.dataset.path)
                save_path = Path(self.config.data_save_dir, f'karate_spread_simple_{graph_idx+1}.pt')
                # save_path_g = Path(config.data_save_dir, f'karate_spread_simple_G_{graph_idx+1}.pkl')
                # nx.write_gp
                # print(type(data))
                torch.save(self.data[-1], save_path)
                
                print(f"Saved graph {graph_idx+1} to {save_path}")

                if self.config.dataset.plot:
                    ax = plt.subplot(ncols, nrows, graph_idx+1)
                    model.draw_network(sources, informed_activations,
                                   spontaneous_activations, used_links, fig=fig, ax=ax)

            self.num_graphs = len(self.data)
            self.num_nodes = self.data[0].num_nodes
            self.num_features = self.data[0].x.shape[1]
            self.num_classes = 0 if self.data[0].y.dim == 1 else int(self.data[0].y.max().item() + 1)
            self.edge_index_shape = self.data[0].edge_index.shape
            
        if self.config.dataset.plot:
            plt.show(block=False)
            plt.pause(1)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class KarateClubSimpleDatasetInductive:
    """
    Dataset for the Karate Club graph.
    Same number of nodes and connectivity in each graph.
    Inductive refers to unseen spread realizations.
    """


    def __init__(self, config: DictConfig, transform=None, pre_transform=None):
        self.config = config
        # super().__init__(None, transform, pre_transform)
        
        # print(self.processed_paths)

        self.num_graphs = None
        self.num_nodes = None
        self.num_features = None
        self.num_classes = None
        self.data_train: List[Data] = []
        self.data_val: List[Data] = []
        self.data_test: List[Data] = []
        self.pos = None
        self.model = None

        
    def process(self):
        save_dir = Path(self.config.data_save_dir)
        # self.processed_dir = save_dir / config.dataset.processed_dir
        
        if self.config.dataset.plot:
            ncols = 5 
            nrows = self.config.dataset.num_graphs // ncols + (self.config.dataset.num_graphs % ncols > 0)
            fig, axs = plt.subplots(ncols, nrows, figsize=(5, 3))
        
        if self.config.dataset.load_data:
            # TODO implement train/val/test split as done in the else branch
            raise NotImplementedError("Not Implemented.")
            filenames = os.listdir(save_dir)
            data_list = [None] * len(filenames)
            found_data = False
            for file_idx, file in enumerate(filenames):
                if file.startswith('karate_spread_simple'):
                    found_data = True
                    # self.data.append(torch.load(save_dir / file))
                    data_list[file_idx] = file
                    
            if not found_data:
                raise FileNotFoundError('No data found')
        
        else:

            # Train/validation/test split
            transform_split = T.RandomNodeSplit(split='random', num_splits=1,
                        num_val=0.2, num_test=0.2, num_train_per_class=10)
            transform_degree_features = T.OneHotDegree(max_degree=20)
            transform_self_loops = T.AddSelfLoops(fill_value=1.0)
            transform_random_walk_pe = T.AddRandomWalkPE(walk_length=self.config.dataset.rwalk_length, attr_name='rwalk_pe')
            transform_virtual_node = T.VirtualNode()
            
            transform = T.Compose([
                # transform_split, # DO NOT USE. SPLIT IS DONE "GRAPH" WISE IN THIS DATASET.
                # transform_degree_features,
                transform_virtual_node,
                transform_random_walk_pe,
                # transform_self_loops
            ])
            
            model = KarateSpreadModel(self.config, p_informed_activation=self.config.dataset.p_informed_activation,
                                    p_spontaneous_activation=self.config.dataset.p_spontaneous_activation)
            self.model = model
            self.transmission_rate_dict = model.transmission_rates
            self.G: nx.Graph = model.G
            
            # TODO Declare these in constructor
            self.num_nodes = self.G.number_of_nodes()
            self.num_edges = self.G.number_of_edges()
            self.num_graphs = self.config.dataset.num_graphs
            self.num_spreads = self.config.dataset.num_graphs

            # source_pool = model.rng.choice(list(model.G.nodes), 
            #     size=self.config.dataset.num_initial_informed*self.config.dataset.num_graphs, replace=False)
            source_pool = [[node] for node in list(model.G.nodes)]
            # source_pool = model.rng.choice(source_pool, size=10, replace=False)
            # source_pool = self.config.dataset.source_pool if self.config.dataset.source_pool is not None else [['1']]
            if self.config.dataset.source_pool != 'None':
                source_pool = self.config.dataset.source_pool
            else:
                source_pool = [model.rng.choice(source_pool, size=model.rng.choice([1,3,6], size=1)).tolist() for i in range(self.config.dataset.n)]
            
            graph_init_id = -1
            
            num_discarded = 0
            num_total_discarded = {tuple(sources): 0 for sources in source_pool}
            
            for initial_node in source_pool:
                graph_init_id += 1
                data_list = []
                graph_id = -1
                num_discarded = 0
                for graph_idx in tqdm(range(self.config.dataset.num_graphs), desc=f'Graph {graph_init_id}', leave=True):
                    graph_id += 1
                    end_activated: torch.Tensor = torch.zeros((len(self.G.nodes), 2), dtype=torch.float)

                    # model = KarateSpreadModel(self.config, p_informed_activation=self.config.dataset.p_informed_activation,
                    #                         p_spontaneous_activation=self.config.dataset.p_spontaneous_activation)
                    # Sample initial informed nodes
                    sources =  initial_node
                    # sources = source_pool[graph_idx*self.config.dataset.num_initial_informed:(graph_idx+1)*self.config.dataset.num_initial_informed]
                    informed_activations, spontaneous_activations, used_links = model.spread(
                        infected=sources)

                    # G: nx.Graph = model.G
                    
                    # Generate data matrix for torch_geometric
                    # y: torch.Tensor = torch.zeros((len(G.nodes), 2), dtype=torch.float)
                    for node in self.G.nodes:
                        # Set node features
                        activated = int(node in informed_activations or node in spontaneous_activations)
                        end_activated[int(node), activated] = 1.0
                        
                    for edge in self.G.edges:
                        # Set edge features
                        self.G.edges[edge]['used_edge'] = int(edge in used_links)
                        self.G.edges[edge]['transmission_rate'] = model.transmission_rates[edge]

                        
                    data: Data = tg.utils.from_networkx(self.G)
                    self.real_edge_index = data.edge_index
                    data.used_links = used_links
                    data.sources = sources
                    data.validate(raise_on_error=True)

                    
                    if self.config.dataset.direction == 'initial_to_end':
                        # data.y = y.squeeze().long()
                        data.y = end_activated.squeeze()
                        data.y_ = torch.max(data.y, 1)[1]
                        data.x = torch.zeros((len(self.G.nodes),1), dtype=torch.float)
                        data.x[int(sources[0])] = 1.0
                        data = transform(data)
                        # data.x = torch.cat([data.x[...,None], data.rwalk_pe.to(data.x.dtype)], dim=-1)
                    elif self.config.dataset.direction == 'end_to_initial':
                        data.x = end_activated.squeeze()[:,1][...,None]

                        # Single output neuron
                        # data.y = torch.zeros((len(self.G.nodes), 1), dtype=torch.float)
                        # data.y[np.array(sources, int)] = 1.0
                        # data.y_ = data.y
                        # data.pos_weight = 1 - data.y.sum()/len(data.y)
                        # # data.pos_weight = torch.tensor(1.0)

                        # Two output neurons
                        data.y = torch.zeros((len(self.G.nodes), 2), dtype=torch.float)
                        data.y[:, 0] = 1.0
                        data.y[np.array(sources, int), 1] = 1.0
                        data.y[np.array(sources, int), 0] = 0.0
                        data.y_ = torch.max(data.y, 1)[1]
                        data.pos_weight = None


                        data.x_ = data.x
                        data = transform(data)
                        data.x = torch.cat([data.x, data.rwalk_pe.to(data.x.dtype)], dim=-1)
                        # data = T.normalize_features.NormalizeFeatures('x')(data)
                        data.x = ((data.x - data.x.mean(0))/(data.x.std(0)+0.0001))

                        # Debug
                        # data.y = data.x
                        # data.y_ = data.y.squeeze().long()
                        # Debug

                        
                        # Discard if no end nodes are activated
                        if data.x[:,0].sum() == 0.:
                            num_total_discarded[tuple(sources)] += 1
                            continue
                        

                    new_edge_index, new_trans = tg.utils.add_self_loops(edge_index=data.edge_index,
                                                edge_attr=data.transmission_rate, fill_value=1.0)
                    data.edge_index = new_edge_index
                    data.transmission_rate = new_trans
                    data.att_weights = torch.ones(data.edge_index.shape[1], dtype=torch.float)
                    data.graph_id = graph_id
                    data.graph_init_id = graph_init_id
                    data.informed_activations = informed_activations
                    data.spontaneous_activations = spontaneous_activations
                    data.sources: List['str'] = sources

                    data_list.append(data)

                    # save_path = hydra.utils.to_absolute_path(config.dataset.path)
                    save_path = save_dir/ f'karate_spread_simple_init_{graph_init_id}_graph_{graph_idx+1}.pt'
                    # save_path_g = Path(config.data_save_dir, f'karate_spread_simple_G_{graph_idx+1}.pkl')
                    # nx.write_gp
                    # print(type(data))
                    torch.save(data_list[-1], save_path)
                    
                    # print(f"Saved initial state {graph_init_id} graph {graph_idx+1} to {save_path}")

                    if self.config.dataset.plot:
                        ax = plt.subplot(ncols, nrows, graph_idx+1)
                        model.draw_network(sources, informed_activations,
                                       spontaneous_activations, used_links, fig=fig, ax=ax)

                
                # Append to train/val/test per one initialization
                # Split data # TODO Get rid of append and 
                self.train_mask = torch.zeros(len(data_list), dtype=torch.bool)
                self.val_mask = torch.zeros(len(data_list), dtype=torch.bool)
                self.test_mask = torch.zeros(len(data_list), dtype=torch.bool)
                self.train_mask[:int(0.6*len(data_list))] = True
                self.val_mask[int(0.6*len(data_list)):int(0.8*len(data_list))] = True
                self.test_mask[int(0.8*len(data_list)):] = True
                self.train_idxs = torch.nonzero(self.train_mask, as_tuple=False)
                self.val_idxs = torch.nonzero(self.val_mask, as_tuple=False)
                self.test_idxs = torch.nonzero(self.test_mask, as_tuple=False)

                self.data_train.extend(data_list[i.item()] for i in self.train_idxs)
                self.data_val.extend(data_list[i.item()] for i in self.val_idxs)
                self.data_test.extend(data_list[i.item()] for i in self.test_idxs)
                # self.data_train.extend(data_list[train_mask])
                # self.data_val.extend(data_list[val_mask])
                # self.data_test.extend(data_list[test_mask])
                # self.data_train.extend(data_list[:int(0.6*len(data_list))])
                # self.data_val.extend(data_list[int(0.6*len(data_list)):int(0.8*len(data_list))])
                # self.data_test.extend(data_list[int(0.8*len(data_list)):])
                
                self.transmission_rate: torch.Tensor = data.transmission_rate.float() # TODO Specify this before loop? Required for plotting sample statistics
                
                if self.config.dataset.compute_sample_statistics:
                    # save_dir = Path(self.config.visualize_dir, 'network_statistis', 'all')
                    save_dir_network_statistics = Path(self.config.visualize_dir, 'network_statistics', 'all')
                    save_dir_network_statistics.mkdir(parents=True, exist_ok=True)
                    self.sample_statistics(data_list=data_list,
                    save_dir = save_dir_network_statistics)
                    # save_dir=save_dir

            print("Discarded graphs: ", num_total_discarded)
            if len(self.data_train) == 0 or len(self.data_val) == 0 or len(self.data_test) == 0:
                raise RuntimeError('Not enough data to split into train/val/test')


            # TODO Initialize all this stuff in the constructor
            self.num_features = self.data_train[0].x.shape[1]
            # self.num_classes = 1 if self.data_train[0].y.dim() == 1 else int(self.data_train[0].y.max().item() + 1)
            # self.num_classes = 1 if self.data_train[0].y.dim() == 1 else 
            self.num_classes = self.data_train[0].y.shape[1]
            # self.edge_index_shape = data.edge_index.shape
            self.edge_index = self.data_train[0].edge_index
            # print("Final:", self.edge_index.shape)
            
            # num_inits is the number of unique initial states
            self.num_inits = graph_init_id + 1

            
            # self.t_rates_norm = tg.utils.softmax(src=self.transmission_rate, index=self.edge_index[1]).float()
            # self.t_rates_norm = tg.utils.softmax(src=self.transmission_rate, index=self.real_edge_index[1]).float()
            # self.t_rates_norm = tg.utils.softmax(src=self.transmission_rate, 
            #     index=np.array(self.g.edges()).astype(int).t[1]).float()
            self.t_rates_global_norm = torch.softmax(self.transmission_rate, dim=0).float()
            
            
        if self.config.dataset.plot:
            # plt.show(block=False)
            # plt.pause(1)
            plt.show(block=True)
            
    def sample_statistics(self, data_list: List[Data], save_dir: Path,
                             num_samples: int = 1000):
        """
        Sample statistics for multiple spreads given a single initial state.
        """
        
        # Compute histogram of infected nodes for each spread in data_list
        node_counts = np.zeros((self.num_nodes))
        for node in self.G.nodes():
            for data in data_list:
                # print(data.informed_activations)
                if node in data.informed_activations:
                    node_counts[int(node)] += 1
                    
        edge_counts = np.zeros((self.num_edges))
        for edge_idx, edge in enumerate(self.G.edges()):
            for data in data_list:
                if edge in data.used_links or edge[::-1] in data.used_links:
                    edge_counts[edge_idx] += 1
                    
        node_counts = node_counts / len(data_list)
        edge_counts = edge_counts / len(data_list)

        save_path = save_dir/f"network_statistics_sources_{data.sources}.png"
        fig, axs = plt.subplots(1,2,figsize=(20,10))
        self.plot_network_statistics(probs=node_counts, edge_probs=edge_counts, sources=data.sources,
                                    fig=fig, axs=axs,
                                    save_path=save_path, title=f'Spread on Karate Club Network (n={self.num_spreads})')
        wandb.log({f"network_statistics_{data.sources}": [wandb.Image(str(save_path))]})
        print("Saved figure for sources", data.sources)
                    
        return node_counts, edge_counts
        
    def plot_network_statistics(self, probs: np.ndarray, sources: List['str'],  save_path: Path,
                                fig, axs, edge_color: np.ndarray = None, edge_probs: np.ndarray = None,
                                edge_attr: torch.Tensor = None, title: str = 'Spread on Karate Club Network'):

        node_color = np.array([
            [1., 0., 0., 0.] if probs[int(node)] != 0 and node not in sources else
            # 'g' if node in data.spontaneous_activations else
            # 'w' if node in source else
            [1., 1., 0., 0.] if node in sources else
            [1., 1., 1., 1.] for node in self.G.nodes()])
        # alpha = np.clip(counts/(np.linalg.norm(counts) + 0.001) + 0.1, 0.0, 1.0)
        node_alpha = np.clip(probs + 0.005, 0.0, 1.0)
        # alpha = np.clip(counts + 0.1, 0.0, 1.0)
        node_alpha[[int(node) for node in sources]] = 1.
        node_color[:,3] += node_alpha
        
        if edge_attr is None and edge_color is None:
            edge_widths: List[float] = [self.transmission_rate_dict[edge]
                           for edge in self.G.edges()]
            r = [1.0, 0.0, 0.0, 0.0]
            b = [0.0, 0.0, 1.0, 0.0]
            edge_color = np.array([
                r if edge_probs[edge_idx] != 0 else
                b for edge_idx, _ in enumerate(self.G.edges())])
            # edge_alphs = np.clip(edge_probs + 0.01, 0.0, 1.0)
            # edge_color[:,3] += edge_alphs
            edge_color[:,3] += 1
        else:
            edge_widths = edge_attr
            # edge_color = ['r' if edge in used_links or (
            #     edge[1], edge[0]) in used_links else 'k' for edge in self.G.edges()]
        nx.draw_networkx(self.G, node_size=200,
                        width=edge_widths,
                        pos=self.model.pos,
                        edge_color=edge_color,
                        node_color=node_color,
                        edgecolors='k', ax=axs[0])

        # Plot histogram of infected nodes
        # Plot big yellow bar at source nodes
        axs[1].bar(np.arange(probs.shape[0]), probs, color=node_color)
        axs[1].set_xticks(np.arange(probs.shape[0]))
        axs[1].set_ylabel("Probability of infection")
        axs[1].set_ylim(0, 1.1)
        axs[1].set_xlabel("Node index")

        fig.suptitle(title)
        
        # Save figure
        # plt.savefig(os.path.join(self.config.visualize_dir, f"network_statistics_sources_{str(sources)}.png"))
        # save_.mkdir(parents=True, exist_ok=True)
        # plt.savefig(save_dir/f"network_statistics_sources_{sources[0]}.png")
        fig.savefig(save_path)
        # plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    def __len__(self):
        return len(self.data_train) + len(self.data_val) + len(self.data_test)
        
    def __getitem__(self, idx):
        return self.data[idx]

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig):

    dataset = KarateClubSimpleDatasetInductive(config)
    loader = DataLoader(dataset.data, batch_size=config.dataset.batch_size, shuffle=True)
    
    print("Loaded")
    
    for data in loader:
        print(data)

if __name__ == "__main__":
    main()
    