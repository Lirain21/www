import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

# to calculate the similarity of the nodes(the support set data and the query set data)
class PointSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        Point Similarity (see paper 3.2.1) Vp_(l-1) -> Ep_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(PointSimilarity, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()] 
        if self.dropout > 0:
            # layer_list += [nn.Dropout2d(p=self.dropout)]
            layer_list += [nn.Dropout(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]
  
        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]
            
        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vp_last_gen, distance_metric):
        """
        Forward method of Point Similarity
        :param vp_last_gen: last generation's node feature of point graph, Vp_(l-1)
        :param ep_last_gen: last generation's edge feature of point graph, Ep_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l) (for Point Loss)
                 l2 version of node similarities
        """

        vp_i = vp_last_gen.unsqueeze(2)    
        vp_j = torch.transpose(vp_i, 1, 2)
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j)**2
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j)
     
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        trans_similarity = torch.exp(-trans_similarity)
    
        ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity)).squeeze(1)
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(vp_last_gen.get_device())

        
        n_query, n1, n2 = ep_ij.size()
        ep_ij_later = ep_ij.reshape(n_query*n1, n2)
        ep_ij_later = F.softmax(ep_ij_later,dim=-1)
        ep_ij_later = ep_ij_later.reshape(n_query, n1, n2)  
        

        ep_ij_later = ep_ij_later*diagonal_mask   

        return ep_ij, ep_ij_later


class P2DAgg(nn.Module):
    def __init__(self, in_c, out_c, dropout = 0.0):
        """
        P2D Aggregation (see paper 3.2.1) Ep_(l) -> Vd_(l)
        :param in_c: number of input channel for the fc layer
        :param out_c:number of output channel for the fc layer
        """
        super(P2DAgg, self).__init__()
        # add the fc layer
        
        self.p2d_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True),
                                             nn.LeakyReLU()])
        self.out_c = out_c
    
        # rewrite the P2D network (2022,10,3)
        '''
        self.in_c = in_c
        self.out_c = out_c
        self.dropout = dropout

        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.out_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.out_c*2),
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.out_c*2, out_channels=self.out_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.out_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        self.point_node_transform = nn.Sequential(*layer_list)
        '''

    def forward(self, point_edge, distribution_node):
        """
        Forward method of P2D Aggregation
        :param point_edge: current generation's edge feature of point graph, Ep_(l)
        :param distribution_node: last generation's node feature of distribution graph, Ed_(l-1)
        :return: current generation's node feature of distribution graph, Vd_(l)
        """
        
        meta_batch = point_edge.size(0) # 25
        num_sample = point_edge.size(1) # 10

        distribution_node_new = torch.cat([point_edge[:, :, :self.out_c], distribution_node], dim=2)
        distribution_node_new = distribution_node_new.view(meta_batch*num_sample, -1)
        distribution_node_new = self.p2d_transform(distribution_node_new)  
        distribution_node_new = distribution_node_new.view(meta_batch, num_sample, -1)

        # distribution_node_new = torch.cat((distribution_node[:,:-1:,:], distribution_node_new[:,-1,:].unsqueeze(1)),dim=1)

        '''
        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(point_edge.get_device())
        edge_feat = F.normalize(point_edge * diag_mask, p=1, dim=-1)
        aggr_feat = torch.bmm(edge_feat, distribution_node)
        node_feat = torch.cat([distribution_node, aggr_feat], -1).transpose(1,2)
        node_feat = self.point_node_transform(node_feat.unsqueeze(-1))
        node_feat = node_feat.transpose(1,2).squeeze(-1)
        '''

        return distribution_node_new
        

class DistributionSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        Distribution Similarity (see paper 3.2.2) Vd_(l) -> Ed_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(DistributionSimilarity, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vd_curr_gen,  distance_metric):
        """
        Forward method of Distribution Similarity
        :param vd_curr_gen: current generation's node feature of distribution graph, Vd_(l)
        :param ed_last_gen: last generation's edge feature of distribution graph, Ed_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l)
        """

        '''
        vd_i = vd_curr_gen.unsqueeze(2)
  
        vd_j = torch.transpose(vd_i, 1, 2)
        if distance_metric == 'l2':
            vd_similarity = (vd_i - vd_j)**2
        elif distance_metric == 'l1':
            vd_similarity = torch.abs(vd_i - vd_j)
    
        trans_similarity = torch.transpose(vd_similarity, 1, 3)
        trans_similarity = torch.exp(-trans_similarity)
        ed_ij = torch.sigmoid(self.point_sim_transform(trans_similarity)).squeeze(1)
        diagonal_mask = 1.0 - torch.eye(vd_curr_gen.size(1)).unsqueeze(0).repeat(vd_curr_gen.size(0), 1, 1).to(vd_curr_gen.get_device())
        n_query, n1, n2 = ed_ij.size()
        ed_ij = ed_ij.reshape(n_query*n1, n2)
        ed_ij = F.softmax(ed_ij, dim=-1)
        ed_ij = ed_ij.reshape(n_query, n1, n2)   
        ed_ij = ed_ij * diagonal_mask
        '''

        # TODO: all the calculation will be in the CPU 
        '''
        n_query, n_sample, n_support = vd_curr_gen.size()
        ed_ij = torch.tr([])
        for i in range(int(n_query)):
            ed_ij_1 = torch.tensor([])
            for j in range(int(n_sample)):
                ed_ij_2 = torch([])
                if j > 0:
                    for y in range(int(j)):
                        simi = (vd_curr_gen[i,y,:]-vd_curr_gen[i,j,:])**2
                        simi = torch.exp(-simi)
                        simi = torch.sum(simi)/int(n_support)
                        # ed_ij_2.append(simi)
                        ed_ij_2 = 
                for z in range(j,int(n_sample)):
                    simi = (vd_curr_gen[i,j,:]-vd_curr_gen[i,z,:])**2
                    simi = torch.exp(-simi)
                    simi = torch.sum(simi)/int(n_support)
                    ed_ij_2.append(simi)
                ed_ij_2 = torch.stack(ed_ij_2, 0)
                ed_ij_1.append(ed_ij_2)
            ed_ij_1 = torch.stack(ed_ij_1, 0)
            ed_ij.append(ed_ij_1)  
        ed_ij = torch.stack(ed_ij, 0)   
        '''
        n_query, n_sample, n_support = vd_curr_gen.size()
        vd_curr_cpu = vd_curr_gen.cpu().detach().tolist()
        ed_ij = []
        for i in range(int(n_query)):
            ed_ij_1 = []
            for j in range(int(n_sample)):
                ed_ij_2 = []
                if j > 0:
                    for y in range(int(j)):
                        simi = (np.array(vd_curr_cpu[i][y])-np.array(vd_curr_cpu[i][j]))**2
                        simi = np.array(simi)
                        simi = np.exp(-simi)
                        simi = (np.sum(simi)/int(n_support)).tolist()
                        ed_ij_2.append(simi)
                for z in range(j,int(n_sample)):
                    simi = (np.array(vd_curr_cpu[i][j]) - np.array(vd_curr_cpu[i][z]))**2
                    simi = np.exp(-simi)
                    simi = (np.sum(simi)/int(n_support)).tolist()
                    ed_ij_2.append(simi)
                ed_ij_1.append(ed_ij_2)
            ed_ij.append(ed_ij_1)

        ed_ij = torch.Tensor(ed_ij).to(vd_curr_gen.get_device())

        diagonal_mask = 1.0 - torch.eye(vd_curr_gen.size(1)).unsqueeze(0).repeat(vd_curr_gen.size(0), 1, 1).to(vd_curr_gen.get_device())
        n_query, n1, n2 = ed_ij.size()
        ed_ij_later = ed_ij.reshape(n_query*n1, n2)
        ed_ij_later = F.softmax(ed_ij_later, dim=-1)
        ed_ij_later = ed_ij_later.reshape(n_query, n1, n2) 

        ed_ij_later = ed_ij_later * diagonal_mask
        return ed_ij, ed_ij_later


class D2PAgg(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        D2P Aggregation (see paper 3.2.2) Ed_(l) -> Vp_(l+1)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(D2PAgg, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        self.point_node_transform = nn.Sequential(*layer_list)

    def forward(self, distribution_edge, point_node):
        """
        Forward method of D2P Aggregation
        :param distribution_edge: current generation's edge feature of distribution graph, Ed_(l)
        :param point_node: last generation's node feature of point graph, Vp_(l-1)
        :return: current generation's node feature of point graph, Vp_(l)
        """
        # get size
        meta_batch = point_node.size(0)
        num_sample = point_node.size(1)

        # get eye matrix (batch_size x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(distribution_edge.get_device())

        # set diagonal as zero and normalize
        edge_feat = F.normalize(distribution_edge * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat, point_node)

        node_feat = torch.cat([point_node, aggr_feat], -1).transpose(1, 2)
        # non-linear transform
        node_feat = self.point_node_transform(node_feat.unsqueeze(-1))
        node_feat = node_feat.transpose(1, 2).squeeze(-1)
     
        return node_feat


class MVRFAN(nn.Module):
    def __init__(self, num_generations, dropout, num_support_sample, num_sample, loss_indicator, point_metric, distribution_metric, device):
        """
        DPGN model
        :param num_generations: number of total generations
        :param dropout: dropout rate
        :param num_support_sample: number of support sample
        :param num_sample: number of sample
        :param loss_indicator: indicator of what losses are using
        :param point_metric: metric for distance in point graph
        :param distribution_metric: metric for distance in distribution graph
        """
        super(MVRFAN, self).__init__()
        self.generation = num_generations
        self.dropout = dropout
        self.num_support_sample = num_support_sample
        self.num_sample = num_sample
        self.loss_indicator = loss_indicator
        self.point_metric = point_metric
        self.distribution_metric = distribution_metric
        self.learnable_para = nn.Parameter(torch.tensor([0.5], device=device))
       
        # node & edge update module can be formulated by yourselves
        P_Sim = PointSimilarity(128, 128, dropout=self.dropout)
       
        fc3 = nn.Sequential(nn.Linear(128+num_support_sample,128+num_support_sample),nn.LeakyReLU(),nn.Dropout(0.2),nn.Linear(128+num_support_sample,2))
        fc1 = nn.Sequential(nn.Linear(128,128),nn.LeakyReLU(),nn.Dropout(0.2),nn.Linear(128,2))
        fc2 = nn.Sequential(nn.Linear(num_support_sample,num_support_sample), nn.LeakyReLU(),nn.Dropout(0.2),nn.Linear(num_support_sample,2 ))
        self.add_module('initial_edge', P_Sim)
        for l in range(self.generation):
            D2P = D2PAgg(128*2, 128, dropout=self.dropout if l < self.generation-1 else 0.0)
            P2D = P2DAgg(2*num_support_sample, num_support_sample)
            P_Sim = PointSimilarity(128, 128, dropout=self.dropout if l < self.generation-1 else 0.0)
            D_Sim = DistributionSimilarity(num_support_sample,
                                            num_support_sample,
                                            dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('point2distribution_generation_{}'.format(l), P2D)
            self.add_module('distribution2point_generation_{}'.format(l), D2P)
            self.add_module('point_sim_generation_{}'.format(l), P_Sim)
            self.add_module('distribution_sim_generation_{}'.format(l), D_Sim)
        self.add_module('classifier', fc1)
        self.add_module('classifier1', fc2)
        self.add_module('classifier2', fc3)

    def forward(self, point_node, distribution_node, simi_A):
        """
        Forward method of DPGN
        :param middle_node: feature extracted from second last layer of Embedding Network
        :param point_node: feature extracted from last layer of Embedding Network
        :param distribution_node: initialized nodes of distribution graph
        :param distribution_edge: initialized edges of distribution graph
        :param point_edge: initialized edge of point graph
        :return: classification result
                 instance_similarity
                 distribution_similarity
        """
        point_similarities = []
        dis_similarities = []
        for l in range(self.generation):
            point_edge , point_edge_softmax= self._modules['point_sim_generation_{}'.format(l)](point_node, self.point_metric)
            point_similarities.append(point_edge * self.loss_indicator[0])
            if l == 0:
                distribution_edge , distribution_edge_softmax = self._modules['distribution_sim_generation_{}'.format(l)](distribution_node, self.distribution_metric)
                dis_similarities.append(distribution_edge)
            point_edge_softmax = point_edge_softmax + self.learnable_para * simi_A
            distribution_node = self._modules['point2distribution_generation_{}'.format(l)](point_edge_softmax, distribution_node)
            distribution_edge , distribution_edge_softmax = self._modules['distribution_sim_generation_{}'.format(l)](distribution_node, self.distribution_metric)
            point_node = self._modules['distribution2point_generation_{}'.format(l)](distribution_edge_softmax, point_node)
            dis_similarities.append(distribution_edge)

        all_node = torch.cat((point_node, distribution_node), dim=-1)
        s_feat_all = all_node[:,:-1,:]
        q_feat_all = all_node[:,-1,:]
        s_logits_all = self._modules['classifier2'](s_feat_all)
        q_logits_all = self._modules['classifier2'](q_feat_all)

        s_feat_p = point_node[:, :-1, :]
        q_feat_p = point_node[:, -1, :]
        s_logits_p = self._modules['classifier'](s_feat_p) # s_logits.size([16,22,2])
        q_logits_p = self._modules['classifier'](q_feat_p)  # q_logits.size([16,2])

        s_feat_d = distribution_node[:,:-1,:]
        q_feat_d = distribution_node[:,-1,:]
        s_logits_d = self._modules['classifier1'](s_feat_d)
        q_logits_d = self._modules['classifier1'](q_feat_d)
        
        logits_dict = {'s_logits_all':s_logits_all,'q_logits_all':q_logits_all,'s_logits_p':s_logits_p,'q_logits_p':q_logits_p,'s_logits_d':s_logits_d,'q_logits_d':q_logits_d}

        return logits_dict, point_similarities, dis_similarities



