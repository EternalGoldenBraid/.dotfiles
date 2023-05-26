from typing import List, Tuple
from pathlib import Path
import glob
import os

import scipy as sp
import mediapy
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm

import wandb



def plot_grad_flow(named_parameters, fig, ax, width=0.2):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            # ave_grads.append(p.cpu().grad.abs().mean())
            # max_grads.append(p.cpu().grad.abs().max())
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())

    ax.bar(np.arange(len(max_grads)), max_grads, width=width, alpha=0.8, lw=1, color="r", label="max_grads")
    ax.bar(np.arange(len(max_grads))+width, ave_grads, width=width, alpha=0.8, lw=1, color="b", label="ave_grads")
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )

    ax.set_xticks(range(0,len(ave_grads), 1), layers, rotation=45)
    ax.set_xlim(left=0, right=len(ave_grads))
    ax.set_ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    
    
def plot_probs2(probs: torch.tensor, gt: torch.tensor,
                ax: plt.axes, title: str):
    # ax.plot(probs[:,0].detach().cpu().numpy(), label='probs', color='blue')
    # ax.plot(dataset.data_train[0].x[:,0].detach().cpu().numpy(), label='x', color='red')
    # ax.plot(dataset.data_train[0].y[:,1].detach().cpu().numpy(), label='GT', color='green')
    ax.scatter(range(probs.shape[0]),
                    probs,
                    label='probs', color='blue', s=20, alpha=0.5)
    # ax.scatter(range(len(dataset.data_train[0].x[:,0].detach().cpu().numpy())),
    #                 dataset.data_train[0].x_.detach().cpu().numpy(),
    #                 label='x', color='red', s=20, alpha=0.8)
    # ax.scatter(range(gt.shape[0]), gt[:,1],
    ax.scatter(range(len(gt)), gt,
                    label='GT', color='green', s=60, alpha=1.0, marker='o', facecolors='none')
    ax.set_xlim(-1, probs.shape[0]+1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title)
    plt.pause(0.1)
                 
def symmetrize_attention(n_nodes: int, edge_index: torch.tensor, att_weights: torch.tensor):
    # Symmetrize attention weights
    A = torch.zeros((att_weights.shape[0], n_nodes, n_nodes), dtype=torch.float32)
    A[:,edge_index[0], edge_index[1]] = att_weights[:].squeeze()
    A = (A + A.transpose(1,2))/2
                 
    return A

def plot_probs_batch(dataset, G: nx.Graph, att_weights: torch.tensor, sources: list[list[str]], probs: torch.tensor,
                edge_index: torch.tensor, axs: plt.axes, fig: plt.figure, title: str, save_dir: str, cmap :plt.cm,
                epoch: int = 0):
 
    # Run inference
    # probs = torch.zeros((dataset.num_nodes+1, 2), dtype=torch.float32)
    # att_weights = torch.zeros((dataset.edge_index.shape[1], 1), dtype=torch.float32)
    G_edges = np.array(G.edges()).astype(int).T

    ### Symmetrize edges
    # edge_attr = symmetrize_attention(
    #                 n_nodes=probs.shape[1], edge_index=edge_index, att_weights=att_weights
    #                 )[:,G_edges[0], G_edges[1]]

    A = torch.zeros((probs.shape[1], probs.shape[1]), dtype=torch.float32)
    A[edge_index[0], edge_index[1]] = att_weights[:].squeeze()
    # A = (A + A.transpose(1,2))/2
    A = (A + A.T)/2
    edge_attr = A[G_edges[0], G_edges[1]]
    
    # Color edges by similarity of edge_attr to true transmission rates from blue to red
    true_edge_weights = torch.tensor(list(dataset.transmission_rate_dict.values()))
                 
    kendall_tau = sp.stats.kendalltau(edge_attr.detach().cpu().numpy(), true_edge_weights.detach().cpu().numpy())
    pears_corr = sp.stats.pearsonr(edge_attr.detach().cpu().numpy(), true_edge_weights.detach().cpu().numpy())

    # Normalize edge_attr
    # edge_attr = edge_attr/edge_attr.max()
    # edge_attr -= edge_attr.min(dim=0,keepdim=True)[0]
    # edge_attr /= (edge_attr.max(dim=0,keepdim=True)[0] - edge_attr.min(dim=0, keepdim=True)[0])
    edge_attr = (edge_attr - edge_attr.min())/(edge_attr.max() - edge_attr.min())
    true_edge_weights = (true_edge_weights - true_edge_weights.min())/(true_edge_weights.max() - true_edge_weights.min())
    # true_edge_weights = true_edge_weights/true_edge_weights.max()

    # Compute similarity
    similarity = torch.cosine_similarity(edge_attr, true_edge_weights, dim=0)
    # similarity = similarity.detach().cpu().numpy()
    distance = torch.abs(edge_attr - true_edge_weights)
    

    # Norm is used to map the similarity to the range [0, 1]
    # edge_color = cmap(norm(distance))
    # edge_color = cmap(distance)
    edge_color = cmap(edge_attr)
    # edge_color = np.ones((edge_attr.shape[0], 4))

    real_nodes = np.array(G.nodes()).astype(int)
    # real_probs = probs[real_nodes, 1].numpy()
    real_probs = probs[:,real_nodes].squeeze().numpy()
    # real_probs = real_probs/real_probs.sum()
    
    title = (f"Sources: {sources}, \
            Number of runs: TODO, similarity: {similarity:.2f}, \
            mean distance: {distance.mean():.2f}, \
            epoch: {epoch}, \
            Kendall tau: {kendall_tau.statistic:.2f}, p-value: {kendall_tau.pvalue:.2f}, \
            Pearson corr: {pears_corr.statistic:.2f}, p-value: {pears_corr.pvalue:.2f} \
            ")
            
    axs[0].cla()
    axs[1].cla()
    dataset.plot_network_statistics(probs=real_probs, title=title, axs=axs, fig=fig,
                sources=sources, save_path=save_dir / f"probs_{sources}_epoch_{epoch}.png",
                edge_attr=edge_attr.numpy(), edge_color=edge_color)

    wandb.log({"val": {
            str(sources) : 
                {"pears_corr": pears_corr.statistic, 
                "kendall_tau": kendall_tau.statistic,
                "pears_corr_pval": pears_corr.pvalue,
                "kendall_tau_pval": kendall_tau.pvalue,
                "pears_corr_abs": np.abs(pears_corr.statistic)}
            }
                })
                 
def plot_probs(config, dataset, G: nx.Graph, att_weights: torch.tensor, sources: list[list[str]], probs: torch.tensor,
                edge_index: torch.tensor, axs: plt.axes, fig: plt.figure, title: str, save_dir: str, cmap :plt.cm,
                epoch: int = 0):
 
    # Run inference
    # probs = torch.zeros((dataset.num_nodes+1, 2), dtype=torch.float32)
    # att_weights = torch.zeros((dataset.edge_index.shape[1], 1), dtype=torch.float32)
    G_edges = np.array(G.edges()).astype(int).T

    ### Symmetrize edges
    edge_attr = symmetrize_attention(
                    n_nodes=probs.shape[1], edge_index=edge_index, att_weights=att_weights
                    )[:,G_edges[0], G_edges[1]]
    
    # Color edges by similarity of edge_attr to true transmission rates from blue to red
    true_edge_weights = torch.tensor(list(dataset.transmission_rate_dict.values()))
                 
    kendall_tau = [sp.stats.kendalltau(attr.detach().cpu().numpy(), true_edge_weights.detach().cpu().numpy()) for attr in edge_attr]
    pears_corr = [sp.stats.pearsonr(attr.detach().cpu().numpy(), true_edge_weights.detach().cpu().numpy()) for attr in edge_attr]

    # Normalize edge_attr
    # edge_attr = edge_attr/edge_attr.max()
    edge_attr -= edge_attr.min(dim=1,keepdim=True)[0]
    edge_attr /= (edge_attr.max(dim=1,keepdim=True)[0] - edge_attr.min(dim=1, keepdim=True)[0])
    true_edge_weights = true_edge_weights/true_edge_weights.max()

    # Compute similarity
    similarity = torch.cosine_similarity(edge_attr, true_edge_weights, dim=1)
    # similarity = similarity.detach().cpu().numpy()
    distance = torch.abs(edge_attr - true_edge_weights)
    

    # Norm is used to map the similarity to the range [0, 1]
    # edge_color = cmap(norm(distance))
    # edge_color = cmap(distance)
    norm = plt.Normalize(vmin=0, vmax=1)
    edge_color = cmap(norm(edge_attr))
    print(edge_color)
    edge_color = cmap(edge_attr)
    print(edge_color)
    # edge_color = np.ones((edge_attr.shape[0], 4))

    real_nodes = np.array(G.nodes()).astype(int)
    # real_probs = probs[real_nodes, 1].numpy()
    real_probs = probs[:,real_nodes].numpy()
    # real_probs = real_probs/real_probs.sum()
    
    for i in range(att_weights.shape[0]):
        # title = f"Sources: {sources}, Number of runs: {len(data_list)}"
        title = (f"Direction: {config.dataset.direction}, \
                Sources: {sources[i]}, \
                Number of runs: TODO, similarity: {similarity[i]:.2f}, \
                mean distance: {distance[i].mean():.2f}, \
                epoch: {epoch}, \
                Kendall tau: {kendall_tau[i].correlation:2f}, p-value: {kendall_tau[i].pvalue:.2f}, \
                Pearson corr: {pears_corr[i][0]:2f}, p-value: {pears_corr[i][1]:.2f} \
                ")
                
        axs[0].cla()
        axs[1].cla()
        dataset.plot_network_statistics(probs=real_probs[i], title=title, axs=axs, fig=fig,
                    sources=sources[i], save_path=save_dir / f"probs_{sources[i]}_epoch_{epoch}.png",
                    edge_attr=edge_attr[i].numpy(), edge_color=edge_color[i])

        wandb.log({"val": {"pears_corr": pears_corr[i][0],
                    "kendall_tau": kendall_tau[i].correlation,
                    "pears_corr_pval": pears_corr[i][1],
                    "kendall_tau_pval": kendall_tau[i].pvalue,
                    "pears_corr_abs": np.abs(pears_corr[i][0]),
                    }})

def make_probs_movie(config, path: Path):
    """
    Make a movie for each initialization of the figures stored in path.
    
    Figure names are of the form: 
        probs_["1","4","5"]_epoch_xxx.png for epoch xxx and some initially infected nodes [1,4,5].
    """

    # Make movie for each initial infection
    # init_infections = {f.split('_')[1] : [] for f in glob.glob(str(path) + '/*') if f.endswith('.png')}
    init_infections = [f.split('_')[1] for f in glob.glob(str(path) + '/*') if f.endswith('.png')]
    init_infections = list(set(init_infections))
    
    out_paths = []

    for init_infection in init_infections:
        print(f"Making movie for {init_infection}")
        
        # This does not work for some reason
        # filenames = glob.glob(str(path) + f'/*{init_infection}*.png')
        # filenames = sorted(filenames)
        all_files = glob.glob(str(path) + f'/*{init_infection}*.png')
        all_files = sorted(all_files)
        search_pattern = str(path) + "/probs_" +str(init_infection)
        filenames = ([f for f in all_files if search_pattern in f])
        filenames = sorted(filenames, key=lambda x: int(''.join(filter(str.isdigit, x))))

        # movie_out = str(path.parent) + f'/movie_{init_infection}.gif'
        movie_out = str(path.parent) + f'/movie_{init_infection}.mp4'
        # Clear movie if it already exists
        if os.path.exists(movie_out):
            os.remove(movie_out)

        # Make gif movie
        # with imageio.get_writer(movie_out, mode='I') as writer:
        #     for idx, filename in tqdm(enumerate(filenames)):
        #         image = imageio.imread(filename)
        #         writer.append_data(image)
        # Make mp4 movie
        # with imageio.get_writer(str(path.parent) + f'/movie_{init_infection}.mp4', mode='I', fps=1) as writer:
        #     for idx, filename in tqdm(enumerate(filenames)):
        #         image = imageio.imread(filename)
        #         writer.append_data(image)
        # print(f"Done making movie for {init_infection}")

        # Make mp4 movie with ffmpeg with the filenames in filenames
        # ffmpeg -framerate 1 -i probs_['1','4','5']_epoch_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
        # cmd = f"ffmpeg -framerate 1 -i {str(path)}/probs_{init_infection}_epoch_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p {movie_out}"
        # os.system(cmd)

        # Make mp4 movie with mediapy with the filenames in filenames
        images = np.array([mediapy.read_image(f) for f in filenames])
        # RGBA to RGB
        images = images[:,:,:,:3]
        mediapy.write_video(movie_out, images, fps=10)
        
        out_paths.append(movie_out)
        
    return out_paths
            
        
def exp_plotting_intervals(n, start=1, end=25):
    """
    Returns n equally spaced points on a log scale between start and end

    :param n: number of epochs
    :param start: starting interval
    :param end: ending interval
    """
    return np.round(np.exp(np.linspace(np.log(start), np.log(end), n))).astype(int)
        
import torch.nn.functional as F
def process_val_batch(fig, axs_probs, val_batch, dataset, val_sources, attention_save_dir, model, cmap_probs, epoch,
                        get_loss_gat_inductive, batch_size):
    with torch.no_grad():
        model.eval()

        batch_val_loss, batch_val_accu, val_logits, edge_index_, att_weights_ = get_loss_gat_inductive(
                                                                                model, batch=val_batch,
                                                                                batch_size=batch_size)

        val_probs = F.softmax(val_logits, 1).detach().cpu()[:,1][None,...]
        att_weights_ = att_weights_.detach().squeeze()

        plot_probs_batch(dataset=dataset, G=dataset.G, att_weights=att_weights_.detach().cpu(),
                   sources=val_sources, probs=val_probs, edge_index=edge_index_.detach().cpu(),
                   axs=axs_probs, fig=fig, title=None, cmap=cmap_probs, epoch=epoch,
                   save_dir=attention_save_dir)

        return val_batch.graph_init_id.item(), batch_val_loss, batch_val_accu, val_probs, att_weights_, edge_index_
    
from functools import partial
def parallel_validation(pool, model, dataset, val_loader, get_loss_gat_inductive,
                        epoch, val_sources, attention_save_dir,
                        figs_probs: List[plt.Figure], axs_probs: List[plt.Axes], cmap_probs):
    num_spreads = 0
    prev_init_id = -1
    val_probs = torch.zeros((dataset.num_inits, dataset.data_val[0].num_nodes))
    att_weights = torch.zeros_like(val_probs)

    # process_batch_partial = partial(process_val_batch, dataset, val_sources, attention_save_dir, model, cmap_probs, epoch ,
    #                                 get_loss_gat_inductive, val_loader.batch_size)
    process_batch_partial = partial(process_val_batch, dataset=dataset, val_sources=val_sources,
                                    attention_save_dir=attention_save_dir, model=model, cmap_probs=cmap_probs,
                                    epoch=epoch, get_loss_gat_inductive=get_loss_gat_inductive,
                                    batch_size=val_loader.batch_size)
    
    # results = []
    # for 
    # TODO wandb and what happens when val_loader is longer than figs_probs: List[num_workers*int]?
    results = list(pool.starmap(process_batch_partial, zip(figs_probs, axs_probs, val_loader)))
    val_loss = 0
    val_accu = 0

    for graph_init_id, batch_val_loss, batch_val_accu, batch_val_probs, batch_att_weights, edge_index_ in results:
        if prev_init_id != graph_init_id:
            if prev_init_id != -1:
                att_weights[prev_init_id] /= num_spreads
                val_probs[prev_init_id] /= num_spreads

            prev_init_id = graph_init_id
            num_spreads = 0

        val_loss += batch_val_loss
        val_accu += batch_val_accu
        val_probs[graph_init_id] += batch_val_probs
        att_weights[graph_init_id] += batch_att_weights

        num_spreads += 1

    val_loss /= len(val_loader)
    val_accu /= len(val_loader)

    return val_loss, val_accu

def backward_hook(module, grad_input, grad_output):
    print(f"MODULE: {module}, GRAD_INPUT: {grad_input}, GRAD_OUTPUT: {grad_output}")
    print()

def hook(grad):
    # print(grad.mean().item(), grad.std().item(), grad.max().item(), grad.min().item())
    print(grad.mean().item())
    print()
    return grad*1e8