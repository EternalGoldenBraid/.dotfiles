# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import glob
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import copy
from pathlib import Path
from time import perf_counter
import pickle

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig
import omegaconf
import hydra
from tqdm import tqdm

# from src.slaps.data_loader import load_data
import sys
from src.slaps.utils import (accuracy, get_random_mask, get_random_mask_ogb,
                            nearest_neighbors, normalize, attention_transmission_metrics,
                            compute_node_clustered_edge_correlations)

from src.data.data_loaders import KarateClubSimpleDatasetInductive as KCSDataset
from src.models.model import GAT, GCN_C, GCN_DAE, MLP, MLP_1

from src.utils import (plot_grad_flow, plot_probs,
                         make_probs_movie, backward_hook, hook,
                         exp_plotting_intervals)

from torch_geometric.loader import DataLoader
import torch_geometric as tg

EOS = 1e-10

import wandb

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
    def get_loss_learnable_adj(self, model, mask, features, labels, Adj):
        """ Loss for label prediction """
        logits = model(features, Adj)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def get_loss_gat(self, model, dataloader, state='train'):
        total_loss = 0
        total_accu = 0

        for graph_idx, data in enumerate(dataloader):
            data = data.to(self.device)
            logits, edge_index_, att_weights = model(data.x, data.edge_index)
            logp = F.log_softmax(logits, 1)

            if state == 'train':
                loss = F.nll_loss(logp[data.train_mask], data.y[data.train_mask], reduction='mean')
                accu = accuracy(logp[data.train_mask], data.y[data.train_mask])
            elif state == 'val':
                loss = F.nll_loss(logp[data.val_mask], data.y[data.val_mask], reduction='mean')
                accu = accuracy(logp[data.val_mask], data.y[data.val_mask])
            elif state == 'test':
                loss = 0.0
                accu = accuracy(logp[data.test_mask], data.y[data.test_mask])

            # total_loss += loss.item()
            total_loss += loss
            total_accu += accu.item()
            
            # Drop self loops from edge_index_ and att_weights
            self.att_weights[graph_idx] = att_weights[edge_index_[0] != edge_index_[1]].squeeze()

        total_loss /= len(dataloader)
        total_accu /= len(dataloader)
        return total_loss, total_accu
     
    def get_loss_gat_inductive(self, model, batch, batch_size=1):
        total_loss = 0
        total_accu = 0

        logits, edge_index_, att_weights = model(batch.x, batch.edge_index)
        probs = F.softmax(logits, 1)
        # logp = F.log_softmax(logits, 1)

        # loss = F.nll_loss(logp, data.y, reduction='mean')
        loss = F.binary_cross_entropy_with_logits(logits, batch.y, reduction='mean', 
                # pos_weight=data.pos_weight
                )
        # accu = (probs>0.5).sum()/probs.shape[0].item()
        accu = (probs>0.5).sum()/probs.shape[0]


        return loss, accu, logits, edge_index_, att_weights

    def get_loss_fixed_adj(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def get_loss_mlp(self, model, dataloader):
        total_loss = 0
        total_accu = 0

        for graph_idx, data in enumerate(dataloader):
            logits = model(data.x)

            loss = F.binary_cross_entropy_with_logits(logits, data.y, reduction='mean', pos_weight=data.pos_weight)

            probs = F.softmax(logits, 1)
            # loss = F.cross_entropy(probs, data.y_, reduction='mean')
            
            # probs = torch.sigmoid(logits)
            # loss = F.binary_cross_entropy(probs, data.y, reduction='mean')

            # probs = F.log_softmax(logits, 1)
            # loss = F.nll_loss(probs, data.y_, reduction='mean')

            accu = accuracy(probs, data.y_)

            total_loss += loss
            total_accu += accu.item()

        total_loss /= len(dataloader)
        total_accu /= len(dataloader)

        return total_loss, total_accu

    def get_loss_adj(self, model, features, feat_ind):
        labels = features[:, feat_ind].float()
        new_features = copy.deepcopy(features)
        new_features[:, feat_ind] = torch.zeros(new_features[:, feat_ind].shape)
        logits = model(new_features)
        loss = F.binary_cross_entropy_with_logits(logits[:, feat_ind], labels, weight=labels + 1)
        return loss

    # def get_loss_masked_features(self, model, features, mask, ogb, noise, loss_t):
    def get_loss_masked_features(self, model, features, mask, noise, loss_t):
        """ Loss for feature prediction/DAE """

        if noise == 'mask':
            masked_features = features * (1 - mask)
        elif noise == "normal":
            noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
            masked_features = features + (noise * mask)

        logits, Adj = model(features, masked_features)
        indices = mask > 0

        if loss_t == 'bce':
            features_sign = torch.sign(features).cuda() * 0.5 + 0.5
            loss = F.binary_cross_entropy_with_logits(logits[indices], features_sign[indices], reduction='mean')
        elif loss_t == 'mse':
            loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
        # else:
        #     masked_features = features * (1 - mask)
        #     logits, Adj = model(features, masked_features)
        #     indices = mask > 0
        #     loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        return loss, Adj

    def train_end_to_end_mlp(self, config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        fig_gradients, ax_gradients = plt.subplots(figsize=(10, 10))
        fig_gradients.tight_layout()
        fig_gradients.suptitle('Gradients')
        
        fig_probs, ax_probs = plt.subplots()
        fig_probs.suptitle('Probabilities')

        dataset = KCSDataset(config=config)
        dataset.process()
        for data in dataset.data_train:
            data = data.to(self.device)
            # data.pos_weight = data.pos_weight.to(self.device)
        for data in dataset.data_val:
            data = data.to(self.device)
            # data.pos_weight = data.pos_weight.to(self.device)
        for data in dataset.data_test:
            data = data.to(self.device)
            # data.pos_weight = data.pos_weight.to(self.device)
        train_loader = DataLoader(dataset.data_train, batch_size=config.dataset.batch_size,
                                    shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset.data_val, 
                                # batch_size=config.dataset.batch_size,
                                batch_size=1,
                                shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset.data_test,
                                batch_size=config.dataset.batch_size,
                                shuffle=True, num_workers=0)
        nfeats = dataset.num_features
        nclasses = dataset.num_classes
        dataset.transmission_rate = dataset.transmission_rate.to(self.device)
        dataset.t_rates_norm = dataset.t_rates_norm.to(self.device)
        dataset.t_rates_global_norm = dataset.t_rates_global_norm.to(self.device)
        
        args = config.model
        for trial in range(args.ntrials):
            model = MLP(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout)
            # model = MLP_1(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
            #                num_layers=args.nlayers, dropout=args.dropout)

            wandb.watch(model, log="all", log_freq=50, log_graph=False)
            # model.register_full_backward_hook(backward_hook)
            for n, p in model.named_parameters():
                # p.register_full_backward_hook(backward_hook)
                # if n == 'layers.0.weight':
                    # p.register_hook(hook)
                p.register_hook(hook)
                print(n)



            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            if torch.cuda.is_available():
                model = model.to(self.device)
                # train_mask = train_mask.cuda()
                # val_mask = val_mask.cuda()
                # test_mask = test_mask.cuda()
                # features = features.cuda()
                # labels = labels.cuda()
                # labels_cpu = labels.cpu().numpy()

            best_val_accu = 0.0
            best_Adj = None

            for epoch in tqdm(range(1, args.epochs + 1), desc='Epoch: ', position=0, leave=True):
                model.train()
                optimizer.zero_grad()

                train_loss, train_accu = self.get_loss_mlp(model, dataloader=train_loader)
                train_loss.backward()
                
                optimizer.step()
                    
                if epoch % config.logging.train_print_interval == 0:
                    print(f"Epoch {epoch:05d} | Train Loss {train_loss.item():.4f} | Train Accuracy {train_accu:.4f}")
                    
                if epoch % config.logging.val_print_interval == 0:
                    with torch.no_grad():
                        model.eval()
                        
                        val_loss, val_accu = self.get_loss_mlp(model, dataloader=val_loader)

                        # DEBUG
                        plot_grad_flow(model.named_parameters(), fig=fig_gradients, ax=ax_gradients)
                        # fig_probs.legend()
                        fig_gradients.savefig(f'outputs/grads/grads_{epoch}.png')
                        ax_gradients.cla()
                        # Clear figure
                        # plt.show()
                         
                        logits = model(
                                    dataset.data_train[0].x,
                                    )
                        probs = F.softmax(logits, dim=1)
                        # print(probs[:,0].detach().cpu().numpy(), torch.linalg.vector_norm(probs[:,1]))
                        title = f'Probabilities, epoch {epoch}, loss {train_loss.item():.4f}, acc {train_accu:.4f}'
                        ax_probs.cla()
                        plot_probs(
                                #    probs=probs[:,0].detach().cpu().numpy(), 
                                   probs=(probs if probs.ndim == 1 else probs[:,1]).detach().cpu().numpy(), 
                                #    probs=probs[:,1].detach().cpu().numpy(), 
                                   gt=(dataset.data_train[0].y if dataset.data_train[0].y.ndim == 1
                                        else dataset.data_train[0].y[:,1]).detach().cpu().numpy(),
                                   ax=ax_probs, title=title, config=config)
                        fig_probs.savefig("probs.png")
                        # DEBUG

                        if val_accu > best_val_accu:
                            
                            # Save best model2 to disk
                            torch.save(model.state_dict(), 
                                       Path(config.data_save_dir, config.model.checkpoint_dir, "best_model_clf_mlp.pt"))

                            best_val_accu = val_accu
                            print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))

                            # test_loss_, test_accu_ = self.get_loss_gat_inductive(model, dataloader=test_loader)
                            # print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))
                            
                            # Store evaluation metrics
                            wandb.log({"val": {"best_val_accu": best_val_accu}}, commit=False)

                            # logp = F.log_softmax(model(features, edge_index), 1).detach().cpu().numpy()
                            # cm = wandb.plot.confusion_matrix(
                            #         y_true=labels_cpu,
                            #         probs=logp,
                            #         class_names=dataset.party_names,
                            #         title="Slaps Confusion Matrix"
                            #         )
                            # pr = wandb.plot.pr_curve(y_true=labels_cpu,
                            #                         y_probas=logp,
                            #                         labels=dataset.party_names,
                            #                         title="Slaps PR Curve"
                            #                         )
                            # roc = wandb.plot.roc_curve(y_true=labels_cpu, y_probas=logp, labels=dataset.party_names,
                            #                             classes_to_plot=None, 
                            #                             title="Slaps ROC Curve"
                            #                             )
                            # wandb.log( {"pr": pr, "cm": cm, 'roc': roc}, commit=False)

                        logs = {
                            "val_acc": val_accu, 
                            "val_loss": val_loss,
                            }
                        # logs = dict(logs, **d)
                        # logs = dict(logs, **sim)
                        # wandb.log({"val": logs},
                        #     # commit=True
                        #     commit=False
                        #     )
                        #### FOR GLOBAL METRICS

                        # data = [[y1, y2] for y1, y2 in zip(self.att_weights, )
                    
                wandb.log({"train": 
                    {"train_loss": train_loss,
                     "train_acc": train_accu,
                    #  "step": epoch,
                     }}, commit=True
                )

    def train_end_to_end(self, config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        dataset, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_vaaliperttu_data_frozen(config)
        dataset = KCSDataset(config)
        loader = DataLoader(dataset, batch_size=config.dataset.batch_size, shuffle=True, num_workers=0)

        # # Visualizer for adjacency matrix
        # if config.visualizer.visualize_adj:
        #     visualizer = Visualizer(figsize=config.visualizer.figsize,
        #                            dataset=dataset, config=config, 
        #                            adj_init = None)
        #     visualizer.save_figure()

        test_accu = []
        validation_accu = []
        added_edges_list = []
        removed_edges_list = []

        args = config.slaps
        for trial in range(args.ntrials):
            model1 = GCN_DAE(nlayers=args.nlayers_adj, in_dim=nfeats, hidden_dim=args.hidden_adj, nclasses=nfeats,
                             dropout=args.dropout1, dropout_adj=args.dropout_adj1,
                             features=features.cpu(), k=args.k, knn_metric=args.knn_metric, i_=args.i,
                             non_linearity=args.non_linearity, normalization=args.normalization, mlp_h=args.mlp_h,
                             mlp_epochs=args.mlp_epochs, gen_mode=args.gen_mode, sparse=args.sparse,
                             mlp_act=args.mlp_act)
            model2 = GCN_C(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                           sparse=args.sparse)
            
            wandb.watch(model1, model2)

            optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr_adj, weight_decay=args.w_decay_adj)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.w_decay)

            if torch.cuda.is_available():
                model1 = model1.cuda()
                model2 = model2.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                labels_cpu = labels.cpu().numpy()

            best_val_accu = 0.0
            best_model2 = None
            best_Adj = None

            for epoch in range(1, args.epochs_adj + 1):
                model1.train()
                model2.train()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                mask = get_random_mask(features, args.ratio, args.nr).cuda()

                # Train denoiser separately before training classifier
                accu = 0
                if epoch < args.epochs_adj // args.epoch_d:
                    model2.eval()
                    # loss1, Adj = self.get_loss_masked_features(model1, features, mask, ogb, args.noise, args.loss)
                    loss1, Adj = self.get_loss_masked_features(model1, features, mask, args.noise, args.loss)
                    loss2 = torch.tensor(0).cuda()
                else:
                    # loss1, Adj = self.get_loss_masked_features(model1, features, mask, ogb, args.noise, args.loss)
                    loss1, Adj = self.get_loss_masked_features(model1, features, mask, args.noise, args.loss)
                    loss2, accu = self.get_loss_learnable_adj(model2, train_mask, features, labels, Adj)
                    

                loss = loss1 * args.lambda_ + loss2
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                

                if epoch % 100 == 0:
                    print("Epoch {:05d} | Train Loss {:.4f}, {:.4f}".format(epoch, loss1.item() * args.lambda_,
                                                                            loss2.item()))
                    if config.slaps.save_adjacency:
                        visualizer.update(Adj.detach().cpu().numpy())
                        visualizer.save_figure()

                # Don't print validation and test accuracy before classifier is trained.
                if epoch >= args.epochs_adj // args.epoch_d and epoch % 1 == 0:
                    with torch.no_grad():
                        model1.eval()
                        model2.eval()

                        val_loss, val_accu = self.get_loss_learnable_adj(model2, val_mask, features, labels, Adj)

                        if val_accu > best_val_accu:
                            
                            # Save best model2 to disk
                            torch.save(model2.state_dict(), 
                                       Path(config.outputs_dir, config.slaps.slaps_save_dir, "best_model_clf.pt"))


                            best_val_accu = val_accu
                            print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))
                            test_loss_, test_accu_ = self.get_loss_learnable_adj(model2, test_mask, features, labels,
                                                                                 Adj)
                            print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))
                            
                            # Store evaluation metrics
                            wandb.log({"val": {"best_val_accu": best_val_accu}}, commit=False)

                            logp = F.log_softmax(model2(features, Adj), 1).detach().cpu().numpy()
                            cm = wandb.plot.confusion_matrix(
                                    y_true=labels_cpu,
                                    probs=logp,
                                    class_names=dataset.party_names,
                                    title="Slaps Confusion Matrix"
                                    )
                            pr = wandb.plot.pr_curve(y_true=labels_cpu,
                                                    y_probas=logp,
                                                    labels=dataset.party_names,
                                                    title="Slaps PR Curve"
                                                    )
                            roc = wandb.plot.roc_curve(y_true=labels_cpu, y_probas=logp, labels=dataset.party_names,
                                                        classes_to_plot=None, 
                                                        title="Slaps ROC Curve"
                                                        )
                            wandb.log( {"pr": pr, "cm": cm, 'roc': roc}, commit=False)
                            
                            # Save adjacency
                            best_Adj = Adj
                            # if config.visualizer.visualize_adj:
                            #     visualizer.save_figure()

                        wandb.log({"val":
                            {"val_acc": val_accu,
                             "val_loss": val_loss,
                            #  "step": epoch,
                             }}, commit=False
                             )
                    
                wandb.log({"train": 
                    {"train_loss": loss,
                     "DAE_loss": loss1,
                     "classifier_loss": loss2,
                     "train_acc": accu,
                    #  "step": epoch,
                     }}, commit=True
                )

            # validation_accu.append(best_val_accu.item())
            # model1.eval()
            # model2.eval()

            # with torch.no_grad():
            #     print("Test Loss {:.4f}, test Accuracy {:.4f}".format(test_loss_, test_accu_))
            #     test_accu.append(test_accu_.item())

        # self.print_results(validation_accu, test_accu)
        # if config.visualizer.visualize_adj:
        #     visualizer.save_figure()

        if config.slaps.save_adjacency and best_Adj is not None:
            save_dir = Path(config.outputs_dir, config.slaps.slaps_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / f"adjacency_best_val_acc_trial_{trial}.pkl", "wb") as f:
                features = pickle.dump(best_Adj.detach().cpu().numpy(), f)
        
        wandb.run.summary["best_val_accuracy"] = best_val_accu
        
    def train_end_to_end_gat_inductive(self, config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        fig_gradients, ax_gradients = plt.subplots(figsize=(20, 10))
        # Clear plot dir
        save_dir_grads = Path(config.visualize_dir, 'gradients')
        save_dir_grads.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(save_dir_grads)+'/*'):
            os.remove(f)
        # fig_gradients.tight_layout()
        fig_gradients.suptitle('Gradients')
        
        fig_probs, axs_probs = plt.subplots(1,2, figsize=(20, 10))
        fig_probs.suptitle('Probabilities')
        # Create color map for edges
        cmap_probs = plt.get_cmap('RdBu')
        norm = plt.Normalize(vmin=0, vmax=1)
        cmapper = lambda x: cmap_probs(norm(x))
        fig_probs.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_probs), ax=axs_probs,
                 shrink=0.5, orientation='horizontal')

        attention_save_dir = Path(config.visualize_dir, 'attention','val','images')
        attention_save_dir.mkdir(parents=True, exist_ok=True)
        # Clear plot dir
        for f in glob.glob(str(attention_save_dir) + '/*'):
            os.remove(f)

        dataset = KCSDataset(config=config)
        dataset.process()
        val_sources = []
        for data in dataset.data_train:
            data = data.to(self.device)
        for data in dataset.data_val:
            data = data.to(self.device)
            # TODO Nasty, but needed for visualizing the probs network with attention weights.
            if data.sources not in val_sources:
                val_sources.append(data.sources)
        for data in dataset.data_test:
            data = data.to(self.device)
        # return
        train_loader = DataLoader(dataset.data_train, batch_size=config.dataset.batch_size,
                                    shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset.data_val, 
                                # batch_size=config.dataset.batch_size,
                                batch_size=1,
                                shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset.data_test,
                                batch_size=config.dataset.batch_size,
                                shuffle=False, num_workers=0)
        nfeats = dataset.num_features
        nclasses = dataset.num_classes
        dataset.transmission_rate = dataset.transmission_rate.to(self.device)
        # dataset.t_rates_norm = dataset.t_rates_norm.to(self.device)
        dataset.t_rates_global_norm = dataset.t_rates_global_norm.to(self.device)
        dataset.edge_index = dataset.edge_index.to(self.device)
        
        self.att_weights = torch.zeros(
                                (dataset.num_inits, dataset.edge_index.shape[1]),
                                dtype=torch.float32, requires_grad=False).to(self.device)
        
        # loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=dataset.pos_weight)

        # # Visualizer for adjacency matrix
        if config.visualizer.visualize_attention:
            from test_vis import AttentionVisualizer
            # visualizer = Visualizer(figsize=config.visualizer.figsize,
            #                        dataset=dataset, config=config, 
            #                    class DataLoader(torch.utils.data.DataLoader):]
            visualizer_attention = AttentionVisualizer(config=config, n_samples=dataset.num_inits)

        args = config.model
        for trial in range(args.ntrials):
            model = GAT(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout, heads=args.nheads,
                           concat=args.concat_multihead, device=self.device, attention_coefs=config.model.attention_coefs)

            wandb.watch(model, log="all", log_freq=50, log_graph=False)
            # model.register_full_backward_hook(backward_hook)
            for n, p in model.named_parameters():
                # p.register_full_backward_hook(backward_hook)
                # p.register_hook(hook)
                pass
            
            if config.model.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start, weight_decay=args.w_decay)
            elif config.model.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=args.w_decay)
                
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_end)

            if torch.cuda.is_available():
                model = model.to(self.device)

            best_val_accu = 0.0
            best_val_loss = torch.inf
            best_Adj = None
            
            plotting_intervals: List[int]  = exp_plotting_intervals(n=args.epochs+1, start=3, end=25)
            for epoch in tqdm(range(1, args.epochs + 1), desc='Epoch: ', position=0, leave=True):
                train_loss = 0.0
                train_accu = 0.0
                val_loss = 0.0
                val_accu = 0.0
                test_loss = 0.0
                test_accu = 0.0
                model.train()
                optimizer.zero_grad()
                
                for train_batch in train_loader:

                    batch_train_loss, batch_train_accu, _, _, _ = self.get_loss_gat_inductive(model, 
                                                                batch=train_batch, batch_size=train_loader.batch_size)
                    
                    batch_train_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_loss += batch_train_loss
                    train_accu += batch_train_accu
                    
                    
                train_loss /= len(train_loader)
                train_accu /= len(train_loader)

                plot_grad_flow(model.named_parameters(), fig=fig_gradients, ax=ax_gradients)
                # plt.show()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                fig_gradients.savefig(save_dir_grads / f'grads_{epoch}.png')
                ax_gradients.cla()
                    
                if epoch == 1 or epoch % config.logging.train_print_interval == 0:
                    print(f"Epoch {epoch:05d} | Train Loss {train_loss:.4f} | Train Accuracy {train_accu:.4f}")
                    
                    
                # if epoch % config.logging.val_print_interval == 0:
                if epoch == 1 or epoch % plotting_intervals[epoch] == 0:
                    
                    # if val_loader.shuffle != False:
                        # raise ValueError("Validation loader shuffle must be False")

                    with torch.no_grad():
                        model.eval()
                        
                        num_spreads = 0
                        prev_init_id = -1
                        val_probs = torch.zeros((dataset.num_inits, dataset.data_val[0].num_nodes))

                        for val_batch in val_loader:
                            
                            # Compute average attention weights per initialization
                            # Assumes that the validation loader is sorted by initialization id
                            # TODO remove this dependency to dataloader
                            if prev_init_id != val_batch.graph_init_id:
                                if prev_init_id != -1:
                                    self.att_weights[prev_init_id] /= num_spreads
                                    val_probs[prev_init_id] /= num_spreads
                                    
                                    # pearson_r = pearsonr(self.att_weights[prev_init_id][dataset.G].detach().cpu().numpy(),
                                                        # )
                                prev_init_id = val_batch.graph_init_id
                                num_spreads = 0 

                            batch_val_loss, batch_val_accu, val_logits, edge_index_, att_weights_ = self.get_loss_gat_inductive(model, batch=val_batch, 
                                                                    batch_size=val_loader.batch_size)

                            val_loss += batch_val_loss
                            val_accu += batch_val_accu

                            # print("DEBUG:", self.att_weights.shape, att_weights_.shape, val_batch.graph_init_id.item())
                            val_probs[val_batch.graph_init_id.item()] += F.softmax(val_logits, 1).detach().cpu()[:,1]
                            self.att_weights[val_batch.graph_init_id.item()] += att_weights_.detach().squeeze()
                            # val_batch.att_weights = att_weights_.detach().squeeze()
                            
                            num_spreads += 1
                            
                        # Plot probs and attention weights
                        plot_probs(dataset=dataset, G=dataset.G, att_weights=self.att_weights.detach().cpu(),
                                sources=val_sources, probs=val_probs, edge_index=edge_index_.detach().cpu(),
                                axs=axs_probs, fig=fig_probs, title=None, cmap=cmap_probs, epoch=epoch,
                                save_dir=attention_save_dir, config=config)
                        
                        val_loss /= len(val_loader)
                        val_accu /= len(val_loader)

                        if val_accu > best_val_accu or val_loss < best_val_loss:
                            
                            # Save best model2 to disk
                            torch.save(model.state_dict(), 
                                       Path(config.data_save_dir, config.model.checkpoint_dir, "best_model_clf.pt"))
                            print("Saved best model to disk at:",
                                 Path(config.data_save_dir, config.model.checkpoint_dir, "best_model_clf.pt"))

                            best_val_accu = val_accu
                            best_val_loss = val_loss
                            print("Val Loss (best) {:.4f} ({:.4f}), Val Accuracy {:.4f}".format(val_loss, best_val_loss, val_accu))

                            # test_loss_, test_accu_ = self.get_loss_gat_inductive(model, dataloader=test_loader)
                            # print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))
                            
                            # Store evaluation metrics
                            wandb.log({"val": {"best_val_accu": best_val_accu}}, commit=False)

                            # logp = F.log_softmax(model(features, edge_index), 1).detach().cpu().numpy()
                            # cm = wandb.plot.confusion_matrix(
                            #         y_true=labels_cpu,
                            #         probs=logp,
                            #         class_names=dataset.party_names,
                            #         title="Slaps Confusion Matrix"
                            #         )
                            # pr = wandb.plot.pr_curve(y_true=labels_cpu,
                            #                         y_probas=logp,
                            #                         labels=dataset.party_names,
                            #                         title="Slaps PR Curve"
                            #                         )
                            # roc = wandb.plot.roc_curve(y_true=labels_cpu, y_probas=logp, labels=dataset.party_names,
                            #                             classes_to_plot=None, 
                            #                             title="Slaps ROC Curve"
                            #                             )
                            # wandb.log( {"pr": pr, "cm": cm, 'roc': roc}, commit=False)

                        
                        #### FOR NODE CLUSTERED METRICS
                        # similarity = compute_node_clustered_edge_correlations(att_weights=self.att_weights.detach(), 
                        #                                                     transmission_rate=dataset.t_rates_norm,
                        #                                                     idx = dataset.edge_index[1])
                        
                        # Similarity is shape (num_initializations, num_nodes) 
                        # Form a dict of form {num_initializations, {node_idx: similarity[init, node_idx]}}
                        # sim = {"val":{
                        #     f"init_{init}": {f"node_{n}" : similarity[init, n].item() for n in range(similarity.shape[1])}
                        #     for init in range(similarity.shape[0])  
                        # }}
                        # # d = {f"similarity_{k}": v.item() for (k, v) in enumerate(similarity)}
                        # print("Max similarity: ", similarity.max().item())

                        # logs = {
                        #     "val_acc": val_accu, 
                        #     "val_loss": val_loss,
                        #     "max_similarity": similarity.max().item(),
                        #     "min_similarity": similarity.min().item(),
                        #     }
                        # # logs = dict(logs, **d)
                        # logs = dict(logs, **sim)
                        # wandb.log({"val": logs},
                        #     # commit=True
                        #     commit=False
                        #     )
                        #### FOR NODE CLUSTERED METRICS
                        
                        #### FOR GLOBAL METRICS
                        # similarity, mse = attention_transmission_metrics(
                        #                     att_matrix=self.att_weights,
                        #                     trans=dataset.t_rates_global_norm
                        #                     )
                        # print("Similarity: ", similarity)
                        # print("MSE: ", mse)
                        # print("Mean att weights: ", self.att_weights.mean(dim=1))
                        # print(self.att_weights.mean(dim=1))

                        logs = {
                            "val_acc": val_accu, 
                            "val_loss": val_loss,
                            # "similarity": {f"init_{init}": similarity[init].item() for init in range(similarity.shape[0])},
                            # "mse": {f"init_{init}": mse[init].item() for init in range(mse.shape[0])},
                            }
                        # logs = dict(logs, **d)
                        # logs = dict(logs, **sim)
                        wandb.log({"val": logs},
                            # commit=True
                            commit=False
                            )
                        #### FOR GLOBAL METRICS

                        # data = [[y1, y2] for y1, y2 in zip(self.att_weights, )
                    
                wandb.log({"train": 
                    {"train_loss": train_loss,
                     "train_acc": train_accu,
                    #  "step": epoch,
                     }}, commit=True
                )

            # validation_accu.append(best_val_accu.item())
            # model1.eval()
            # model2.eval()

            # with torch.no_grad():
            #     print("Test Loss {:.4f}, test Accuracy {:.4f}".format(test_loss_, test_accu_))
            #     test_accu.append(test_accu_.item())

        # self.print_results(validation_accu, test_accu)
        # if config.visualizer.visualize_adj:
        #     visualizer.save_figure()

        # if config.slaps.save_adjacency and best_Adj is not None:
        #     save_dir = Path(config.outputs_dir, config.slaps.slaps_save_dir)
        #     save_dir.mkdir(parents=True, exist_ok=True)
        #     with open(save_dir / f"adjacency_best_val_acc_trial_{trial}.pkl", "wb") as f:
        #         features = pickle.dump(best_Adj.detach().cpu().numpy(), f)
        
        wandb.run.summary["best_val_accuracy"] = best_val_accu
        video_save_paths = make_probs_movie(path=attention_save_dir, config=config)
        wandb.log({
            f"video_{i}": wandb.Video(path, fps=4, format="mp4") for i, path in enumerate(video_save_paths)
            })

    def train_end_to_end_gat_transductive(self, config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # # Visualizer for adjacency matrix
        # if config.visualizer.visualize_adj:
        #     visualizer = Visualizer(figsize=config.visualizer.figsize,
        #                            dataset=dataset, config=config, 
        #                            adj_init = None)
        #     visualizer.save_figure()

        dataset = KCSDatasetTrans(config=config)
        dataset.process()
        loader = DataLoader(dataset, batch_size=config.dataset.batch_size, shuffle=True, num_workers=0)
        nfeats = dataset.num_features
        nclasses = dataset.num_classes
        
        self.att_weights = torch.zeros(
                                (len(dataset), dataset.edge_index_shape[1]),
                                dtype=torch.float32)

        test_accu = []
        validation_accu = []
        added_edges_list = []
        removed_edges_list = []

        args = config.model
        for trial in range(args.ntrials):
            model = GAT(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout, heads=args.nheads,
                           concat=args.concat_multihead)

            wandb.watch(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            if torch.cuda.is_available():
                model = model.to(self.device)
                # train_mask = train_mask.cuda()
                # val_mask = val_mask.cuda()
                # test_mask = test_mask.cuda()
                # features = features.cuda()
                # labels = labels.cuda()
                # labels_cpu = labels.cpu().numpy()

            best_val_accu = 0.0
            best_val_loss = torch.inf
            best_Adj = None

            for epoch in tqdm(range(1, args.epochs + 1), desc='Epoch: ', position=0, leave=True):
                model.train()
                optimizer.zero_grad()

                train_loss, train_accu = self.get_loss_gat(model, dataloader=loader, state='train')
                train_loss.backward()
                optimizer.step()
                    
                if epoch % 2 == 0:
                    print(f"Epoch {epoch:05d} | Train Loss {train_loss.item():.4f}")
                    with torch.no_grad():
                        model.eval()
                        
                        val_loss, val_accu = self.get_loss_gat(model, dataloader=loader, state='val')

                        if val_accu > best_val_accu or val_loss < best_val_loss:
                            
                            # Save best model2 to disk
                            torch.save(model.state_dict(), 
                                       Path(config.data_save_dir, config.model.checkpoint_dir, "best_model_clf.pt"))

                            best_val_accu = val_accu
                            best_val_loss = val_loss
                            print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))

                            # test_loss_, test_accu_ = self.get_loss_gat(model, dataloader=loader, state='test')
                            # print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))
                            
                            # Store evaluation metrics
                            wandb.log({"val": {"best_val_accu": best_val_accu}}, commit=False)

                            # logp = F.log_softmax(model(features, edge_index), 1).detach().cpu().numpy()
                            # cm = wandb.plot.confusion_matrix(
                            #         y_true=labels_cpu,
                            #         probs=logp,
                            #         class_names=dataset.party_names,
                            #         title="Slaps Confusion Matrix"
                            #         )
                            # pr = wandb.plot.pr_curve(y_true=labels_cpu,
                            #                         y_probas=logp,
                            #                         labels=dataset.party_names,
                            #                         title="Slaps PR Curve"
                            #                         )
                            # roc = wandb.plot.roc_curve(y_true=labels_cpu, y_probas=logp, labels=dataset.party_names,
                            #                             classes_to_plot=None, 
                            #                             title="Slaps ROC Curve"
                            #                             )
                            # wandb.log( {"pr": pr, "cm": cm, 'roc': roc}, commit=False)
                            
                        wandb.log({"val":
                            {"val_acc": val_accu,
                             "val_loss": val_loss,
                            #  "step": epoch,
                             }}, commit=False
                             )
                    
                wandb.log({"train": 
                    {"train_loss": train_loss,
                     "train_acc": train_accu,
                    #  "step": epoch,
                     }}, commit=True
                )

            # validation_accu.append(best_val_accu.item())
            # model1.eval()
            # model2.eval()

            # with torch.no_grad():
            #     print("Test Loss {:.4f}, test Accuracy {:.4f}".format(test_loss_, test_accu_))
            #     test_accu.append(test_accu_.item())

        # self.print_results(validation_accu, test_accu)
        # if config.visualizer.visualize_adj:
        #     visualizer.save_figure()

        # if config.slaps.save_adjacency and best_Adj is not None:
        #     save_dir = Path(config.outputs_dir, config.slaps.slaps_save_dir)
        #     save_dir.mkdir(parents=True, exist_ok=True)
        #     with open(save_dir / f"adjacency_best_val_acc_trial_{trial}.pkl", "wb") as f:
        #         features = pickle.dump(best_Adj.detach().cpu().numpy(), f)
        
        wandb.run.summary["best_val_accuracy"] = best_val_accu

    def print_results(self, validation_accu, test_accu):
        print(test_accu)
        print("std of test accuracy", np.std(test_accu))
        print("average of test accuracy", np.mean(test_accu))
        print(validation_accu)
        print("std of val accuracy", np.std(validation_accu))
        print("average of val accuracy", np.mean(validation_accu))

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    
    # os.environ[“CUDA_VISIBLE_DEVICES”] = ‘,’.join(str(x) for x in args.gpu)
    
    # if torch.cuda.device_count() > 1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    experiment = Experiment()
    
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    os.environ["WANDB_MODE"] = config.wandb.wandb_mode
    
    run = wandb.init(
        project=config.wandb.project_name,
        config = wandb.config,
        )
    

    if config.model.name == "end2end":
        experiment.train_end_to_end(config)
    elif config.model.name == "end2end_mlp":
        experiment.train_end_to_end_mlp(config)
    elif config.model.name == "gat_inductive":
        experiment.train_end_to_end_gat_inductive(config)
    elif config.model.name == "gat_transductive":
        experiment.train_end_to_end_gat_transductive(config)
    else:
        raise ValueError("Model not supported")


if __name__ == '__main__':

    main()
