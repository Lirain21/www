import random
import os
# from matplotlib.pyplot import flag
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger

from sklearn.datasets import fetch_species_distributions
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class Meta_Trainer(nn.Module):
    def __init__(self, args, model):
        super(Meta_Trainer, self).__init__()

        self.args = args

        self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss().to(args.device)

        self.num_loss_generation = args.num_loss_generation
        self.generation_weight = args.generation_weight

        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query

        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step

        self.trial_path = args.trial_path
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0 
        
        self.res_logs=[]

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples=samples.to(self.device)
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)
            
            '''
            y_label = s_data.y
            y_len = s_data.y.size(0)
            y_len = int(y_len/2)
            s_y_neg = y_label[:y_len].unsqueeze(0)
            s_y_pos = y_label[y_len:].unsqueeze(0)
            s_neg = torch.tensor([0]).unsqueeze(0).to(s_y_neg.device)
            s_pos = torch.tensor([1]).unsqueeze(0).to(s_y_neg.device)
            s_data.y = torch.cat((s_y_neg, s_neg, s_y_pos, s_pos),1).squeeze(0)
            '''

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            s_data = self.loader_to_samples(s_data)
           
            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
           
            q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)

            # add the s_data super node label: 0 anad 1
            '''
            y_label = s_data.y
            y_len = s_data.y.size(0)
            y_len = int(y_len/2)
            s_y_neg = y_label[:y_len].unsqueeze(0)
            s_y_pos = y_label[y_len:].unsqueeze(0)
            s_neg = torch.tensor([0]).unsqueeze(0).to(s_y_neg.device)
            s_pos = torch.tensor([1]).unsqueeze(0).to(s_y_neg.device)
            s_data.y = torch.cat((s_y_neg, s_neg, s_y_pos, s_pos),1).squeeze(0)
            '''
            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

        return adapt_data, eval_data

    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        frel_0 = lambda x: x[0] == 'relationnet' and '0' in x[1]
        frel_1 = lambda x: x[0] == 'relationnet' and '1' in x[1]
        fclf = lambda x: x[0] == 'relationnet' and 'classifier' in x[1]
        fclf1 = lambda x:x[0] == 'relationnet' and 'classifier1' in x[1]
        fclf2 = lambda x:x[0] == 'relationnet' and 'claddifier2' in x[1]
        if adapt_weight==0:
            flag=lambda x: not fenc(x)
        elif adapt_weight==1:
            flag=lambda x: not (fenc(x) or frel_0(x) or frel_1(x))     
        else:
            flag= lambda x: True
        if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
            adaptable_weights = None
        else:
            adaptable_weights = []
            adaptable_names=[]
            for name, p in model.module.named_parameters():
                names=name.split('.')
                if flag(names):
                    adaptable_weights.append(p)
                    adaptable_names.append(name)
        return adaptable_weights
              
    def get_loss(self, model, batch_data, logits_dict, point_similarity, dis_similarity, train=True, flag = 0, epoch=0, task_id=0, use_d_loss = 1):
        n_support_train = self.args.n_shot_train # 10
        n_support_test = self.args.n_shot_test   # 10
        n_query = self.args.n_query # 16

        s_logits_p = logits_dict['s_logits_p']
        q_logits_p = logits_dict['q_logits_p']
        s_logits_d = logits_dict['s_logits_d']
        q_logits_d = logits_dict['q_logits_d']
        s_logits_all = logits_dict['s_logits_all']
        q_logits_all = logits_dict['q_logits_all']

        # add the disagreement loss 
        # dis_loss = F.mse_loss(s_logits_p, s_logits_d) + F.mse_loss(q_logits_p, q_logits_d)

        if not train:       
            losses_adapt_p = self.criterion(s_logits_p.reshape(2*(n_support_test)*n_query,2), batch_data['s_label'].repeat(n_query))
            losses_adapt_d = self.criterion(s_logits_d.reshape(2*(n_support_test)*n_query,2), batch_data['s_label'].repeat(n_query))
            losses_adapt_all = self.criterion(s_logits_all.reshape(2*(n_support_test)*n_query,2), batch_data['s_label'].repeat(n_query))
        else:
            if flag:
                losses_adapt_p = self.criterion(s_logits_p.reshape(2*(n_support_train)*n_query,2), batch_data['s_label'].repeat(n_query)) 
                losses_adapt_d = self.criterion(s_logits_d.reshape(2*(n_support_train)*n_query,2), batch_data['s_label'].repeat(n_query))   
                losses_adapt_all = self.criterion(s_logits_all.reshape(2*(n_support_train)*n_query,2), batch_data['s_label'].repeat(n_query))   

  
            else:
                losses_adapt_p = self.criterion(q_logits_p, batch_data['q_label'])
                losses_adapt_d = self.criterion(q_logits_d, batch_data['q_label'])
                losses_adapt_all = self.criterion(q_logits_all, batch_data['q_label'])
        
        if torch.isnan(losses_adapt_p).any() or torch.isinf(losses_adapt_p).any():
            print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt_p)
            print(s_logits_p)
            losses_adapt_p = torch.zeros_like(losses_adapt_p)
        if self.args.reg_adj > 0:
            # n_support = 22
            n_support = batch_data['s_label'].size(0)
            # add the super node 
            
            adj_p = point_similarity[-1]
            adj_d = dis_similarity[-1]
            if train:
                if flag:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    n_d = n_query * (n_support)
                    n_d_1 = n_query * n_support

                    label_edge = model.label2edge(s_label).reshape((n_d_1, -1))
                    simi_A_truth = ((model.label2edge(s_label).squeeze(0)))
                    pred_edge_d = adj_d[:,:-1,:-1].reshape((n_d_1, -1))

                    pred_edge_p = adj_p[:,:-1,:-1].reshape((n_d, -1))
                    simi_A_1 = (label_edge)
                    '''
                    if epoch == 4500:
                        print('this is the adj_p_1')
                        print(point_similarity[0][0,:11,:11])
                        print('this is the adj_p_2')
                        print(point_similarity[1][0,:11,:11])
                       
                        print('this is the adj_d_1')
                        print(dis_similarity[0][0,:11,:11])
                        print('this is the adj_d_2')
                        print(dis_similarity[1][0,:11,:11])

                        print('this this the prior similarity')
                        print(simi_A_truth[0,0,:11,:11])
                    '''
                    
                else:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    q_label = batch_data['q_label'].unsqueeze(1)
                    total_label = torch.cat((s_label, q_label), 1)
                    # TODO debug
                    ground_truth_label=model.label2edge(total_label).squeeze(1).squeeze(0)
                    label_edge = model.label2edge(total_label).squeeze(1)[:,-1,:-1] # [16,22,23]

                    # label_edge = model.label2edge(total_label).squeeze(1)[:,-1,:-1] # [16,22,23]
                    pred_edge_d = adj_d[:,-1,:-1] 

                    pred_edge_p = adj_p[:,-1,:-1]
                    simi_A_1 = (label_edge)

                    '''
                    if epoch == 4890 and task_id == 8:
                        adj_p_visi_0 = adj_p[0].squeeze(0).cpu().detach().numpy()
                        adj_d_visi_0 = adj_d[0].squeeze(0).cpu().detach().numpy()
                        adj_p_visi_1 = adj_p[1].squeeze(0).cpu().detach().numpy()
                        adj_d_visi_1 = adj_d[1].squeeze(0).cpu().detach().numpy()
                        adj_p_visi_2 = adj_p[-1].squeeze(0).cpu().detach().numpy()
                        adj_d_visi_2 = adj_d[-1].squeeze(0).cpu().detach().numpy()
                        total_label_visi = ground_truth_label[0].cpu().detach().numpy()
                        p_0 = np.array(adj_p_visi_0)
                        p_1 = np.array(adj_p_visi_1)
                        p_2 = np.array(adj_p_visi_2)
                        d_0 = np.array(adj_d_visi_0)
                        d_1 = np.array(adj_d_visi_1)
                        d_2 = np.array(adj_d_visi_2)

                        c = np.array(total_label_visi)
                        
                        fig, ax = plt.subplots(figsize=(9,9))
                        
                        sns.heatmap(pd.DataFrame(np.round(d_0,2),columns = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9'], index = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9']),
                                                     annot=False, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="Blues_r") # Blues
                        plt.savefig('/workspace2/lrf/www_tripleview_visi_1/results/task8_adj_d_0.png')
                        
                        sns.heatmap(pd.DataFrame(np.round(d_1,2),columns = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9'], index = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9']),
                                                     annot=False, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="Blues_r") # Blues
                        plt.savefig('/workspace2/lrf/www_tripleview_visi_1/results/task8_adj_d_1.png')
                    
                        sns.heatmap(pd.DataFrame(np.round(d_2,2),columns = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9'], index = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9']),
                                                     annot=False, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="Blues_r") # Blues
                        plt.savefig('/workspace2/lrf/www_tripleview_visi_1/results/task8_adj_d_2.png')
                    
                        sns.heatmap(pd.DataFrame(np.round(p_0,2),columns = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9'], index = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9']),
                                                     annot=False, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="Blues_r") # Blues
                        plt.savefig('/workspace2/lrf/www_tripleview_visi_1/results/task8_adj_p_0.png')
                        
                        sns.heatmap(pd.DataFrame(np.round(p_1,2),columns = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9'], index = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9']),
                                                     annot=False, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="Blues_r") # Blues
                        plt.savefig('/workspace2/lrf/www_tripleview_visi_1/results/task8_adj_p_1.png')
                    
                        sns.heatmap(pd.DataFrame(np.round(p_2,2),columns = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9'], index = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9']),
                                                     annot=False, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="Blues_r") # Blues
                        plt.savefig('/workspace2/lrf/www_tripleview_visi_1/results/task8_adj_p_2.png')

                        sns.heatmap(pd.DataFrame(np.round(c,2),columns = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9'], index = ['mol1', 'mol2', 'mol3','mol4','mol5','mol6','mol7','mol8','mol9']),
                                                     annot=False, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="Blues_r") # Blues
                        plt.savefig('/workspace2/lrf/www_tripleview_visi_1/results/task8_adj_groud_truth.png')
                    '''

                    
                    

            else:
                s_label = batch_data['s_label'].unsqueeze(0)
                n_d = n_support # 22
                n_d_1 = n_support
                label_edge = model.label2edge(s_label).reshape((n_d_1, -1))
                pred_edge_d = adj_d[:, :n_support, :n_support].mean(0).reshape((n_d_1, -1))
                pred_edge_p = adj_p[:, :n_support, :n_support].mean(0).reshape((n_d, -1))
                # simi_A = simi_A[:,:n_support,:n_support].mean(0).reshape((n_d, -1))
                simi_A_1 = (label_edge)

            adj_loss_val_p = F.mse_loss(pred_edge_p, simi_A_1)
            # adj_loss_val_d = F.mse_loss(pred_edge_d, label_edge)
            adj_loss_val_d = F.mse_loss(pred_edge_d, simi_A_1)
            
            if torch.isnan(adj_loss_val_p).any() or torch.isinf(adj_loss_val_p).any():
                print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val_p)
                adj_loss_val_p = torch.zeros_like(adj_loss_val_p)

            if use_d_loss:
                losses_adapt_p += self.args.reg_adj * adj_loss_val_d
                losses_adapt_d += self.args.reg_adj * adj_loss_val_d
                losses_adapt_all += self.args.reg_adj * adj_loss_val_d


            losses_adapt_p += self.args.reg_adj * adj_loss_val_p
            losses_adapt_d += self.args.reg_adj * adj_loss_val_p
            losses_adapt_all += self.args.reg_adj * adj_loss_val_p
        
        return losses_adapt_p, losses_adapt_d, losses_adapt_all

    def train_step(self):
        self.train_epoch += 1
        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches={}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db

        for k in range(self.update_step):
            losses_eval_p = []
            losses_eval_d = []
            losses_eval_catpd = []
           
            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                model = self.model.clone()
                model.train()
                adaptable_weights = self.get_adaptable_weights(model)
                
                for inner_step in range(self.inner_update_step):
                    logits_dict,  point_similarity, dis_similarity = model(train_data)
                    total_loss_p, total_loss_d, total_loss_catpd =self.get_loss(model, train_data, logits_dict, point_similarity, dis_similarity, train=True, flag= 1 ,epoch=self.train_epoch, task_id = task_id, use_d_loss = 0)
                    total_loss = total_loss_p + total_loss_d + total_loss_catpd
                    model.adapt(total_loss, adaptable_weights = adaptable_weights)
                
                logits_dict, point_similarity, dis_similarity = model(train_data)

                '''
                if self.train_epoch == 4890 and task_id == 8:
                    print(train_data['s_data']['smiles'])
                '''
                loss_eval_p, loss_eval_d, loss_eval_cat = self.get_loss(model, train_data, logits_dict, point_similarity, dis_similarity, train=True, flag = 0, epoch=self.train_epoch, task_id=task_id, use_d_loss=0)
                
                losses_eval_p.append(loss_eval_p)
                losses_eval_d.append(loss_eval_d)
                losses_eval_catpd.append(loss_eval_cat)
                
            losses_eval_p = torch.stack(losses_eval_p)
            losses_eval_d = torch.stack(losses_eval_d)
            losses_eval_catpd = torch.stack(losses_eval_catpd) 

            losses_eval_p = torch.sum(losses_eval_p)
            losses_eval_d = torch.sum(losses_eval_d)
            losses_eval_catpd = torch.sum(losses_eval_catpd)

            losses_eval_p = losses_eval_p / len(task_id_list)
            losses_eval_d = losses_eval_d / len(task_id_list)   
            losses_eval_catpd = losses_eval_catpd / len(task_id_list)            
            self.optimizer.zero_grad()
            losses_eval_p.backward(retain_graph = True)
            losses_eval_d.backward(retain_graph = True)
            losses_eval_catpd.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            losses_eval = (losses_eval_p+losses_eval_d+losses_eval_catpd)/3
            print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())

        return self.model.module

    def test_step(self):
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            model = self.model.clone()
            if self.update_step_test>0:
                model.train()     
                for i, batch in enumerate(adapt_data['data_loader']):
                    batch = batch.to(self.device)
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                        'q_data': batch, 'q_label': None}
                    adaptable_weights = self.get_adaptable_weights(model)
             
                    logits_dict, point_similarity, dis_similarity = model(cur_adapt_data)
                    total_loss_p, total_loss_d, total_loss_catpd = self.get_loss(model, cur_adapt_data, logits_dict, point_similarity, dis_similarity, train=False, epoch=0, task_id=task_id, use_d_loss = 0)
                    total_loss = (total_loss_p + total_loss_d + total_loss_catpd)/3
                    model.adapt(total_loss, adaptable_weights=adaptable_weights)
                    if i>= self.update_step_test-1:
                        break

            model.eval()
            with torch.no_grad():
                # TODO 记得把数据拼接, 要把y_label and y_prediction cat 
                s_data = eval_data['s_data'].to(self.device)
                q_loader = eval_data['data_loader']
                s_label = eval_data['s_label'].to(self.device)
                
                y_true_list = []
                q_logits_list_p = []
                q_logits_list_d = []
                q_logits_list_catpd = []
 
                for q_data in q_loader:
                    q_data = q_data.to(self.device)
                    q_label = q_data.y
                    logits_dict,_,_= model.forward_query_loader(s_data, q_data)
                    
                    # 实现多数投票
                    # TODO samble learning 
                    y_true_list.append(q_label)
                    
                    q_logits_list_p.append(logits_dict['q_logits_p'])
                    q_logits_list_d.append(logits_dict['q_logits_d'])
                    q_logits_list_catpd.append(logits_dict['q_logits_all'])

             
                q_logits_p = torch.cat(q_logits_list_p, 0)
                q_logits_d = torch.cat(q_logits_list_d, 0)
                q_logits_all = torch.cat(q_logits_list_catpd, 0)
                y_true = torch.cat(y_true_list, 0)
                y_score_p = F.softmax(q_logits_p, dim=-1).detach()[:,1]
                y_score_d = F.softmax(q_logits_d, dim=-1).detach()[:,1]
                y_score_catpd = F.softmax(q_logits_all, dim=-1).detach()[:,1]

                if self.args.eval_support:
                    y_s_score = F.softmax(logits_dict['s_logits_p'],dim=-1).detach()[:,1]
                    y_s_true = s_label
                    y_score=torch.cat([y_score, y_s_score])
                    y_true=torch.cat([y_true, y_s_true])
                auc_p = auroc(y_score_p, y_true, pos_label=1).item()
                auc_d = auroc(y_score_d, y_true, pos_label=1).item()
                auc_all = auroc(y_score_catpd, y_true, pos_label=1).item()
                auc = [auc_p, auc_d, auc_all]
                auc = max(auc)
            auc_scores.append(auc)   

            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc,avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4),)
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        return self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
