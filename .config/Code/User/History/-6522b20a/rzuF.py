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
from time import perf_counter as pc
import pickle
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import torch_geometric
from omegaconf import DictConfig
import omegaconf
import hydra

from tqdm import tqdm


from slaps.data_loader import load_data
from slaps.model import GCN, GCN_C, GCN_DAE, MLP, GAT
from slaps.utils import accuracy, get_random_mask, get_random_mask_ogb, nearest_neighbors, normalize
from slaps.data_loader import load_vaaliperttu_data, load_vaaliperttu_data_frozen
from data.dataloader_slaps_batch_mode import VaalikoneDataset

from utils.utils import (EarlyStopper, aggregate_statistics, aggregate_from_disk, save_user_feature,
                        ConcatenateThenShuffleSampler,
                        )
from slaps.bert_encoder import BertEncoder


from visualizer import Visualizer

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
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def get_loss_fixed_adj(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
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

    # def half_val_as_train(self, val_mask, train_mask):
    #     val_size = np.count_nonzero(val_mask)
    #     counter = 0
    #     for i in range(len(val_mask)):
    #         if val_mask[i] and counter < val_size / 2:
    #             counter += 1gen_mode ==
    #             val_mask[i] = False
    #             train_mask[i] = True
    #     return val_mask, train_mask

    def train_end_to_end_mlp(self, config):
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        

        model_bert = BertEncoder(config, device=self.device)
        tokenizer = model_bert.tokenizer

        full_dataset = VaalikoneDataset(config=config, debug=config.debug, tokenizer=tokenizer)
        
        
        splits = full_dataset.get_data_split()
        
        train_dataset = VaalikoneDataset(config, splits["train"])
        val_dataset = VaalikoneDataset(config, splits["val"])
        test_dataset = VaalikoneDataset(config, splits["test"])
        
        # Create users_data for each split
        train_users_data = train_dataset.create_users_data()
        val_users_data = val_dataset.create_users_data()
        test_users_data = test_dataset.create_users_data()
        

        # Create samplers with users_data and bert_input
        batch_size = config.bert.batch_size
        bert_input = config.bert["max_token_length"]
        train_sampler = ConcatenateThenShuffleSampler(train_users_data, bert_input)
        val_sampler = ConcatenateThenShuffleSampler(val_users_data, bert_input)
        test_sampler = ConcatenateThenShuffleSampler(test_users_data, bert_input)
        
        # Create DataLoaders with custom samplers
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
        
        test_accu = []
        validation_accu = []
        
        itr = iter(train_dataloader)
        batch = next(itr)

        args = config.slaps
        for trial in range(args.ntrials):
        # with Pool(processes=os.cpu_count()-3) as pool:


            nfeats = model_bert.nfeats
            model = MLP(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                           sparse=args.sparse)
            
            wandb.watch(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            optimizer_bert = AdamW(model_bert.model.parameters(), lr=args.lr)
            total_steps = len(dataset) * args.epochs_adj
            scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=5, num_training_steps=total_steps)
            # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.training.warmup_steps, num_training_steps=total_steps)

            if torch.cuda.is_available():
                model = model.to(self.device)
                model_bert = model_bert.to(self.device)
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                # features = torch.empty((len(dataset), nfeats), requires_grad=True).to(self.device)
                participant_feature_mask = torch.tensor(dataset.data.participant_id.values, dtype=int).to(self.device)
                participant_ids = torch.tensor(dataset.participants.participant_id.values).to(self.device)
                # participant_features = torch.empty((dataset.n_participants, nfeats*1), requires_grad=True).to(self.device)
                labels = dataset.party_label.to(self.device)
                labels_cpu = labels.cpu().numpy()

            best_val_accu = 0.0
            best_model2 = None
            best_Adj = None
            
            ### DEBUG
            # input_ids = torch.randint(1024,(len(dataset), 2*64)).to(self.device)
            # attention_mask = torch.randint_like(input_ids, 1).to(self.device).bool()
            ### 
            
            total_time_bert = 0.0
            total_time_saveing_bert = 0.0
            total_time_load_bert = 0.0
            total_time_mlp = 0.0

            for epoch in tqdm(range(1, args.epochs_adj + 1), leave=False, desc="Epochs", position=0):
                model_bert.train()
                model.train()

                optimizer.zero_grad()
                optimizer_bert.zero_grad()

                # mask = get_random_mask(features, args.ratio, args.nr).cuda()

                # Train denoiser separately before training classifier
                accu = torch.tensor(0.0)
                
                count = 0 
                for data in tqdm(loader, leave=False, desc="Batches", position=1):
                    count += 1
                    input_ids = data[0].to(self.device)
                    attention_mask = data[1].to(self.device)
                    # party_id = data[2].to(self.device)
                    participant_ids_ = data[3].to(self.device)
                    features_idxs = data[4].to(self.device)
                    
                    time_bert = pc()
                    bert_features = model_bert(input_ids=input_ids, 
                                                        attention_mask=attention_mask
                                                    ).pooler_output
                                                    # )
                    time_bert = pc() - time_bert
                    
                    detached_features = [f.detach().cpu() for f in bert_features]

                    time_saving_bert = pc()
                    pool.map( save_user_feature, zip(
                                participant_ids_.tolist(),
                                # bert_features,
                                detached_features,
                                [bert_outputs_dir] * bert_features.shape[0]
                                )
                            )
                    time_saving_bert = pc() - time_saving_bert
                time_bert /= count
                time_saving_bert /= count
                    
                
                time_load_bert = pc()
                features = torch.stack(pool.map(aggregate_from_disk, 
                                    zip(dataset.participants.participant_id.values, [bert_outputs_dir] * dataset.n_participants))
                                    ).to(self.device)
                time_load_bert = pc() - time_load_bert
                # total_time_load_bert += time_load_bert

                time_mlp = pc()
                loss, accu = self.get_loss_fixed_adj(model, train_mask, features, labels)
                time_mlp = pc() - time_mlp

                loss.backward()
                optimizer.step()
                optimizer_bert.step()
                scheduler_bert.step()

                wandb.log({"time":
                    {"time_bert": total_time_bert,
                    "time_saveing_bert": total_time_saveing_bert,
                    "time_load_bert": total_time_load_bert,
                    "time_mlp": total_time_mlp,
                    }
                    }, step=epoch, commit=False)
                
                if epoch % 1 == 0:
                    print("Epoch {:05d} | Train Loss {:.4f}, Train Accuracy {:.4f}".format(epoch, loss.item(), accu.item()))  
                    print(f"Time bert: {time_bert}, time saving bert: {time_saving_bert}, time load bert: {time_load_bert}, time mlp: {time_mlp}")
                    
                    
                with torch.no_grad():
                    model.eval()

                    val_loss, val_accu = self.get_loss_fixed_adj(model, val_mask, features, labels)

                    if val_accu > best_val_accu:
                        
                        # Save best model2 to disk
                        torch.save(model.state_dict(), 
                                   Path(config.outputs_dir, config.slaps.slaps_save_dir, "best_model_mlp.pt"))
                        
                        best_val_accu = val_accu
                        print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))
                        test_loss_, test_accu_ = self.get_loss_fixed_adj(model, test_mask, features, labels)
                        # print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))
                        
                        # Store evaluation metrics
                        wandb.log({
                            "val": {"best_val_accu": best_val_accu},
                            "test": {"test_accu": test_accu},
                            }, step=epoch, commit=False)

                        logp = F.log_softmax(model(features), 1).detach().cpu().numpy()
                        cm = wandb.plot.confusion_matrix(
                                y_true=labels_cpu,
                                probs=logp,
                                class_names=dataset.party_names ,
                                title= "MLP Confusion Matrix"
                                )
                        pr = wandb.plot.pr_curve(y_true=labels_cpu,
                                                y_probas=logp,
                                                labels=dataset.party_names,
                                                title="MLP PR Curve"
                                                )
                        roc = wandb.plot.roc_curve(y_true=labels_cpu, y_probas=logp, labels=dataset.party_names,
                                                    classes_to_plot=None,
                                                    title="MLP ROC Curve"
                                                    )
                        wandb.log( {"pr": pr, "cm": cm, 'roc': roc}, step=epoch, commit=False)
                    
                        

                    wandb.log({"val":
                        {"val_acc": val_accu,
                         "val_loss": val_loss,
                         "step": epoch,
                         }}, step=epoch, commit=False
                         )
                    
                wandb.log({"train": 
                    {"train_loss": loss,
                     "train_acc": accu,
                     "step": epoch,
                     }}, step=epoch, commit=True
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

        dataset, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_vaaliperttu_data_frozen(config)
        
        min_delta = 5.0 if config.slaps.gen_mode == 1 else 0.5
        early_stopper = EarlyStopper(patience=config.slaps.patience, 
                        min_delta=min_delta
                        )


        # Visualizer for adjacency matrix
        if config.visualizer.visualize_adj:
            visualizer = Visualizer(figsize=config.visualizer.figsize,
                                   dataset=dataset, config=config, 
                                   adj_init = None)
            visualizer.save_figure()

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
            
            if config.slaps.classifier == 'gcn':
                model2 = GCN_C(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                           sparse=args.sparse)
            
            elif config.slaps.classifier == 'gat':
                model2 = GAT(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                               num_layers=args.nlayers, dropout=args.dropout2, heads=args.nheads,
                               concat=args.concat_multihead, device=None)
            else:
                raise NotImplementedError("Invalid classifier.")
            
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
                    
                    if config.slaps.classifier == 'gat':
                        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj=Adj)
                        loss2, accu = self.get_loss_learnable_adj_gat(model2, train_mask, features, labels, 
                                        edge_index=edge_index, edge_attr=edge_attr)
                    elif config.slaps.classifier == 'gcn':
                        loss2, accu = self.get_loss_learnable_adj(model2, train_mask, features, labels, Adj)
                    else:
                        raise NotImplemented("Invalid classifier.")
                    
                    # Adj_identity = torch.diag(torch.ones(Adj.shape[0], device=Adj.device))
                    # loss2, accu = self.get_loss_learnable_adj(model2, train_mask, features, labels, Adj_identity)
                    

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
                # if epoch >= args.epochs_adj // args.epoch_d and epoch % 10 == 0:
                if epoch >= args.epochs_adj_warmup and epoch % 10 == 0:
                    with torch.no_grad():
                        model1.eval()
                        model2.eval()

                        if config.slaps.classifier == 'gat':
                            val_loss, val_accu = self.get_loss_learnable_adj_gat(model2, val_mask, features, labels, 
                                            edge_index=edge_index, edge_attr=edge_attr)
                        elif config.slaps.classifier == 'gcn':
                            val_loss, val_accu = self.get_loss_learnable_adj(model2, val_mask, features, labels, Adj)
                        else:
                            raise NotImplemented("Invalid classifier.")

                        if val_accu > best_val_accu:
                            
                            # Save best model2 to disk
                            torch.save(model2.state_dict(), 
                                       Path(config.outputs_dir, config.slaps.slaps_save_dir, "best_model_clf.pt"))
                            if config.slaps.classifier == 'gat':
                                logp = F.log_softmax(model2(features, edge_index, edge_attr), 1).detach().cpu().numpy()
                            elif config.slaps.classifier == 'gcn':
                                logp = F.log_softmax(model2(features, Adj), 1).detach().cpu().numpy()
                            else:
                                raise NotImplemented("Invalid classifier.")
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
                            if config.visualizer.visualize_adj:
                                visualizer.save_figure()

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
        if config.visualizer.visualize_adj:
            visualizer.save_figure()

        if config.slaps.save_adjacency and best_Adj is not None:
            save_dir = Path(config.outputs_dir, config.slaps.slaps_save_dir)
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
    
    experiment.train_end_to_end_mlp(config)


    # if config.slaps.model == "end2end":
    #     experiment.train_end_to_end(config)
    # elif config.slaps.model == "2step":
    #     raise NotImplementedError
    #     experiment.train_two_steps()
    # elif config.slaps.model == "knn_gcn":
    #     raise NotImplementedError
    #     experiment.train_knn_gcn()
    # elif config.slaps.model == "end2end_mlp":
    #     experiment.train_end_to_end_mlp(config)


if __name__ == '__main__':

    main()
