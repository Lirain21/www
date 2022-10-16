import os
import json
import numpy as np
import pandas as pd
import torch

def init_trial_path(args,is_save=True):
    """Initialize the path for a hyperparameter setting

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with the trial path updated
    """
    prename = args.dataset + '_' + str(args.test_dataset)+ '_' +str(args.n_shot_test) + '_' + args.enc_gnn
    result_path = os.path.join(args.result_path, prename)
    os.makedirs(result_path, exist_ok=True)
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = result_path + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args.trial_path = path_to_results
    os.makedirs(args.trial_path)
    if is_save:
        save_args(args)

    return args

def save_args(args):
    args = args.__dict__
    json.dump(args, open(args['trial_path'] + '/args.json', 'w'))
    prename=f"upt{args['update_step']}-{args['inner_lr']}_mod{args['batch_norm']}"
    # prename+=f"-{args['rel_hidden_dim']}-{args['rel_res']}"
    json.dump(args, open(args['trial_path'] +'/'+prename+ '.json', 'w'))

def count_model_params(model):
    print(model)
    param_size = {}
    cnt = 0
    for name, p in model.named_parameters():
        k = name.split('.')[0]
        if k not in param_size:
            param_size[k] = 0
        p_cnt = 1
        for j in p.size():
            p_cnt *= j
        param_size[k] += p_cnt
        cnt += p_cnt
    for k, v in param_size.items():
        print(f"Number of parameters for {k} = {round(v / 1024, 2)} k")
    print(f"Total parameters of model = {round(cnt / 1024, 2)} k")

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        self.fpath = fpath
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')
        
    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers, verbose=True):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()
        if verbose:
            self.print()

    def print(self):
        log_str = ""
        for name, num in self.numbers.items():
            log_str += f"{name}: {num[-1]}, "
        print(log_str)

    def conclude(self, avg_k=3):
        avg_numbers={}
        best_numbers={}
        valid_name=[]
        for name, num in self.numbers.items():
            best_numbers[name] = np.max(num)
            avg_numbers[name] = np.mean(num[-avg_k:])
            if str.isdigit(name.split('-')[-1]):
                valid_name.append(name)
        vals=np.array([list(avg_numbers.values()),list(best_numbers.values())])
        cols = list(self.numbers.keys())
        rows = ['avg','best']
        df = pd.DataFrame(vals,index=rows, columns=cols)
        df['mid'] = df[valid_name].apply(lambda x: np.median(x),axis=1)
        df['mean'] = df[valid_name].apply(lambda x: np.mean(x),axis=1)
        save_path = self.fpath +'stats.csv'
        df.to_csv(save_path,sep='\t',index=True)
        return df

    def close(self):
        if self.file is not None:
            self.file.close()

#TODO add the function of the relationnet relational function

def preprocessing(num_supports, num_samples, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set edge mask (to distinguish support and query edges)
    support_edge_mask = torch.zeros(num_samples, num_samples).to(device)
    support_edge_mask[:num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(num_samples, num_samples).to(device)

    return  query_edge_mask, evaluation_mask

# to transpose in support_label,query_label,
def initialize_nodes_edges(s_data, q_data, num_supports, num_queries, num_ways, device):
    """
    :param batch: data batch
    :param num_supports: number of samples in support set
    :param tensors: initialized tensors for holding data
    :param batch_size: how many tasks per batch
    :param num_queries: number of samples in query set
    :param num_ways: number of classes for each few-shot task
    :param device: the gpu device that holds all data

    :return: data of support set,
             label of support set,
             data of query set,
             label of query set,
             data of support and query set,
             label of support and query set,
             initialized node features of distribution graph (Vd_(0)),
             initialized edge features of point graph (Ep_(0)),
             initialized edge_features_of distribution graph (Ed_(0))
    """
   

    # initialize nodes of distribution graph
    # node_gd_init_support.size([16,22,22])
    # node_gd_init_query.size([16,1,22])

    num_batch = int(q_data.y.size(0))
    num_supports = int(s_data.y.size(0))
    node_gd_init_support = label2edge(num_batch,s_data['y'], device)    
    node_gd_init_query = (torch.ones([num_batch,1, num_supports])
                          * torch.tensor(1. / num_supports)).to(device)
    # node_feature_gd.size([38,22])
    node_feature_gd = torch.cat([node_gd_init_support, node_gd_init_query], dim=1)
    '''
    s_label = s_data['y'].unsqueeze(0).repeat(num_batch,1)
    q_label = q_data['y'].unsqueeze(0).repeat()
    all_label = torch.cat([s_label, q_label], 1)
    all_label = all_label.squeeze(0)
    all_label_in_edge = label2edge(all_label, device)
    
    edge_feature_gp = all_label_in_edge.clone()

    # uniform initialization for point graph's edges
    edge_feature_gp[num_supports:, :num_supports] = 1. / num_supports

    edge_feature_gp[:num_supports, num_supports:] = 1. / num_supports

    edge_feature_gp[num_supports:, num_supports:] = 0
    for i in range(num_queries): # 5
        edge_feature_gp[num_supports + i, num_supports + i] = 1

    # initialize edges of distribution graph (same as point graph)
    all_label_in_edge = all_label_in_edge.unsqueeze(0)
    node_feature_gd = node_feature_gd.unsqueeze(0)
    edge_feature_gp = edge_feature_gp.unsqueeze(0)
    edge_feature_gd = edge_feature_gp.clone()
    '''
    return  node_feature_gd

def label2edge(num_batch,label, device):
    """
    convert ground truth labels into ground truth edges
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth edges
    :return: ground truth edges
    """
    label = label.unsqueeze(0).repeat(num_batch,1)
    num_samples = label.size(1)
    # reshape label_i.size([25,5,5]) label_j.size([25,5,5])
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)
    # compute edge edge.size([25,5,5])
    edge = torch.eq(label_i, label_j).float().to(device)
    return edge

def one_hot_encode(num_classes, class_idx, device):
    """
    one-hot encode the ground truth
    :param num_classes: number of total class
    :param class_idx: belonging class's index
    :param device: the gpu device that holds the one-hot encoded ground truth label
    :return: one-hot encoded ground truth label
    """
    return torch.eye(num_classes)[class_idx].to(device)
