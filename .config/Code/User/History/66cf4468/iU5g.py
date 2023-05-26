# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import copy
from pathlib import Path
from time import perf_counter
import pickle

import numpy as np
import torch_geometric
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig
import omegaconf
import hydra

from src.slaps.model import GCN, GCN_C, GCN_DAE, MLP, GAT
from src.slaps.utils import accuracy, get_random_mask

from create_dataset import create_slaps_dataset

from src.utils.utils import EarlyStopper
# from visualizer import Visualizer

EOS = 1e-10

import wandb

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def get_loss_learnable_adj_gat(self, model, mask, features, labels, edge_index, edge_attr):
        """ Loss for label prediction """
        logits = model(x=features, edge_index=edge_index, edge_attr=edge_attr)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def get_loss_learnable_adj(self, model, mask, features, labels, Adj):
        """ Loss for label prediction """
        logits = model(features, Adj)
        # loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], reduction='mean')
        # accu = accuracy(logp[mask], labels[mask])
        accu = loss.item()
        return loss, accu

    # def get_loss_learnable_adj(self, model, mask, features, labels, Adj):
    #     """ Loss for label prediction """
    #     logits = model(features, Adj)
    #     logp = F.log_softmax(logits, 1)
    #     loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
    #     accu = accuracy(logp[mask], labels[mask])
    #     return loss, accu

    # def get_loss_fixed_adj(self, model, mask, features, labels):
    #     logits = model(features)
    #     logp = F.log_softmax(logits, 1)
    #     loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
    #     accu = accuracy(logp[mask], labels[mask])
    #     return loss, accu

    def get_loss_fixed_adj(self, model, mask, features, labels):
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], reduction='mean')

        # accu = accuracy(logp[mask], labels[mask])
        accu = loss.item()
        return loss, accu

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

        data_root = hydra.utils.to_absolute_path(config.data.dataset.root)

        dataset, features, labels, train_mask, val_mask, test_mask = create_slaps_dataset(config)
        nfeats = features.shape[1]
        nclasses = dataset[0].y.shape[0]

        test_accu = []
        validation_accu = []

        model_save_path = Path(config.paths.outputs_dir, config.model.slaps_save_dir, "best_model__mlp_clf.pt")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        args = config.model
        for trial in range(args.ntrials):
            model = MLP(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout)
            
            wandb.watch(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start, weight_decay=args.w_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_end)

            if torch.cuda.is_available():
                model = model.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                labels_cpu = labels.cpu().numpy()

            best_val_accu = 0.0
            best_model2 = None
            best_Adj = None

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()

                # mask = get_random_mask(features, args.ratio, args.nr).cuda()

                # Train denoiser separately before training classifier
                accu = 0
                loss, accu = self.get_loss_fixed_adj(model, train_mask, features, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                

                if epoch % config.training== 0 or epoch == 1:
                    print("Epoch {:05d} | Train Loss {:.4f}".format(epoch, loss.item()))
                # Don't print validation and test accuracy before classifier is trained.
                with torch.no_grad():
                    model.eval()

                    val_loss, val_accu = self.get_loss_fixed_adj(model, val_mask, features, labels)

                    if val_accu > best_val_accu:
                        
                        # Save best model2 to disk
                        torch.save(model.state_dict(), model_save_path)
                                   
                        
                        best_val_accu = val_accu
                        print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))
                        test_loss_, test_accu_ = self.get_loss_fixed_adj(model, test_mask, features, labels)
                        print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))
                        
                        # Store evaluation metrics
                        # wandb.log({
                        #     "val": {"best_val_accu": best_val_accu},
                        #     "test": {"test_accu": test_accu},
                        #     }, commit=False)

                        # logp = F.log_softmax(model(features), 1).detach().cpu().numpy()
                        # cm = wandb.plot.confusion_matrix(
                        #         y_true=labels_cpu,
                        #         probs=logp,
                        #         class_names=dataset.party_names ,
                        #         title= "MLP Confusion Matrix"
                        #         )
                        # pr = wandb.plot.pr_curve(y_true=labels_cpu,
                        #                         y_probas=logp,
                        #                         labels=dataset.party_names,
                        #                         title="MLP PR Curve"
                        #                         )
                        # roc = wandb.plot.roc_curve(y_true=labels_cpu, y_probas=logp, labels=dataset.party_names,
                        #                             classes_to_plot=None,
                        #                             title="MLP ROC Curve"
                        #                             )
                        # wandb.log( {"pr": pr, "cm": cm, 'roc': roc}, commit=False)
                        

                    wandb.log({"val":
                        {"val_acc": val_accu,
                         "val_loss": val_loss,
                        #  "step": epoch,
                         }}, commit=False
                         )
                    
                wandb.log({"train": 
                    {"train_loss": loss,
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
        wandb.run.summary["best_val_accuracy"] = best_val_accu

    def train_end_to_end(self, config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        data_root = hydra.utils.to_absolute_path(config.data.dataset.root)

        dataset, features, labels, train_mask, val_mask, test_mask = create_slaps_dataset(config)
        nfeats = features.shape[1]
        nclasses = dataset[0].y.shape[0]
        
        min_delta = 5.0 if config.model.gen_mode == 1 else 0.5
        early_stopper = EarlyStopper(patience=config.model.patience, 
                        min_delta=min_delta
                        )


        # Visualizer for adjacency matrix
        # if config.visualizer.visualize_adj:
        #     visualizer = Visualizer(figsize=config.visualizer.figsize,
        #                            dataset=dataset, config=config, 
        #                            adj_init = None)
        #     visualizer.save_figure()

        test_accu = []
        validation_accu = []
        added_edges_list = []
        removed_edges_list = []
        
        model_save_path = Path(config.paths.outputs_dir, config.model.slaps_save_dir, "best_model_clf.pt")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        args = config.model
        for trial in range(args.ntrials):
            model1 = GCN_DAE(nlayers=args.nlayers_adj, in_dim=nfeats, hidden_dim=args.hidden_adj, nclasses=nfeats,
                             dropout=args.dropout1, dropout_adj=args.dropout_adj1,
                             features=features.cpu(), k=args.k, knn_metric=args.knn_metric, i_=args.i,
                             non_linearity=args.non_linearity, normalization=args.normalization, mlp_h=args.mlp_h,
                             mlp_epochs=args.mlp_epochs, gen_mode=args.gen_mode, sparse=args.sparse,
                             mlp_act=args.mlp_act)
            
            if config.model.classifier == 'gcn':
                model2 = GCN_C(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                           sparse=args.sparse)
            
            elif config.model.classifier == 'gat':
                model2 = GAT(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                               num_layers=args.nlayers, dropout=args.dropout2, heads=args.nheads,
                               concat=args.concat_multihead, device=None)
            else:
                raise NotImplementedError("Invalid classifier.")
            
            wandb.watch(model1, model2)

            optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr_adj_start, weight_decay=args.w_decay_adj)
            scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.epochs, eta_min=args.lr_adj_end)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr_start, weight_decay=args.w_decay)
            scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=args.lr_end)

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
            
            val_loss = torch.inf

            for epoch in range(1, args.epochs_adj + 1):
                model1.train()
                model2.train()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                mask = get_random_mask(features, args.ratio, args.nr).cuda()

                # Train denoiser separately before training classifier
                accu = 0
                # if epoch < args.epochs_adj // args.epoch_d:
                if epoch < args.epochs_adj_warmup:
                    model2.eval()
                    # loss1, Adj = self.get_loss_masked_features(model1, features, mask, ogb, args.noise, args.loss)
                    loss1, Adj = self.get_loss_masked_features(model1, features, mask, args.noise, args.loss)
                    loss2 = torch.tensor(0).cuda()
                else:
                    # loss1, Adj = self.get_loss_masked_features(model1, features, mask, ogb, args.noise, args.loss)
                    loss1, Adj = self.get_loss_masked_features(model1, features, mask, args.noise, args.loss)
                    
                    if config.model.classifier == 'gat':
                        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj=Adj)
                        loss2, accu = self.get_loss_learnable_adj_gat(model2, train_mask, features, labels, 
                                        edge_index=edge_index, edge_attr=edge_attr)
                    elif config.model.classifier == 'gcn':
                        loss2, accu = self.get_loss_learnable_adj(model2, train_mask, features, labels, Adj)
                    else:
                        raise NotImplemented("Invalid classifier.")
                    
                    # Adj_identity = torch.diag(torch.ones(Adj.shape[0], device=Adj.device))
                    # loss2, accu = self.get_loss_learnable_adj(model2, train_mask, features, labels, Adj_identity)
                    

                loss = loss1 * args.lambda_ + loss2
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                scheduler1.step()
                scheduler2.step()
                

                if epoch % 100 == 0:
                    print("Epoch {:05d} | Train Loss {:.4f}, {:.4f}".format(epoch, loss1.item() * args.lambda_,
                                                                            loss2.item()))
                    # if config.model.save_adjacency:
                    #     visualizer.update(Adj.detach().cpu().numpy())
                    #     visualizer.save_figure()

                # Don't print validation and test accuracy before classifier is trained.
                # if epoch >= args.epochs_adj // args.epoch_d and epoch % 10 == 0:
                if epoch >= args.epochs_adj_warmup and epoch % 10 == 0:
                    with torch.no_grad():
                        model1.eval()
                        model2.eval()

                        if config.model.classifier == 'gat':
                            val_loss, val_accu = self.get_loss_learnable_adj_gat(model2, val_mask, features, labels, 
                                            edge_index=edge_index, edge_attr=edge_attr)
                        elif config.model.classifier == 'gcn':
                            val_loss, val_accu = self.get_loss_learnable_adj(model2, val_mask, features, labels, Adj)
                        else:
                            raise NotImplemented("Invalid classifier.")

                        if val_accu > best_val_accu:
                            
                            # Save best model2 to disk
                            torch.save(model2.state_dict(), model_save_path)
                                       

                            best_val_accu = val_accu
                            print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))
                            # test_loss_, test_accu_ = self.get_loss_learnable_adj(model2, test_mask, features, labels,
                                                                                #  Adj)
                            # print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))
                            
                            # Store evaluation metrics
                            wandb.log({"val": {"best_val_accu": best_val_accu}}, commit=False)

                            if config.model.classifier == 'gat':
                                logp = F.log_softmax(model2(features, edge_index, edge_attr), 1).detach().cpu().numpy()
                            elif config.model.classifier == 'gcn':
                                logp = F.log_softmax(model2(features, Adj), 1).detach().cpu().numpy()
                            else:
                                raise NotImplemented("Invalid classifier.")
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
                            
                            # Save adjacency
                            best_Adj = Adj
                            # if config.visualizer.visualize_adj:
                            #     visualizer.save_figure()

                        if early_stopper.early_stop(val_loss):             
                            print("Early stopping at:")
                            print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))
                            wandb.log({"val":
                                {"val_acc": val_accu,
                                 "val_loss": val_loss,
                                #  "step": epoch,
                                 }}, commit=False
                                 )
                            break

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

        if config.model.save_adjacency and best_Adj is not None:
            save_dir = Path(config.paths.outputs_dir, config.model.slaps_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / f"adjacency_best_val_acc_trial_{trial}.pkl", "wb") as f:
                features = pickle.dump(best_Adj.detach().cpu().numpy(), f)
        
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    experiment = Experiment()
    
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    os.environ["WANDB_MODE"] = config.wandb.wandb_mode
    
    run = wandb.init(
        entity=config.wandb.entity, 
        project=config.wandb.project_name,
        config = wandb.config,
        )
    
    # This script is for  frozen bert
    config.bert.freeze = 'full'

    if config.slaps.model == "end2end":
        experiment.train_end_to_end(config)
    elif config.slaps.model == "end2end_mlp":
        experiment.train_end_to_end_mlp(config)


if __name__ == '__main__':

    main()
