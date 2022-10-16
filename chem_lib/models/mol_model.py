import torch
import torch.nn as nn

from .encoder import GNN_Encoder
from .relation import MLP,ContextMLP
from .mvrfan import MVRFAN
from .fingerprint import calculate_similarity
from ..utils import preprocessing, initialize_nodes_edges

class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


class ContextAwareRelationNet(nn.Module):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()
        self.gpu_id = args.gpu_id
        self.device = args.device
        self.n_support = args.support_num
        self.n_query = args.n_query
        self.edge_type = args.rel_adj
        self.edge_activation = args.rel_act
        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)
        self.delinear = nn.Linear(300, 128)

        
        self.encode_projection = ContextMLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                batch_norm=args.batch_norm,dropout=args.map_dropout,
                                pre_fc=args.map_pre_fc,ctx_head=args.ctx_head)
        

        inp_dim = args.map_dim

        self.relationnet = MVRFAN(
                      num_generations=args.num_generations,
                      dropout=args.dpgn_dropout,
                      num_support_sample=args.support_num,
                      num_sample = args.support_num + args.n_query,
                      loss_indicator=args.loss_indicator,
                      point_metric=args.point_metric,
                      distribution_metric=args.distribution_metric,
                      device = args.device
                      )


    # 将label编辑成edge
    def label2edge(self, label, mask_diag=True):
        num_samples = label.size(1)
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float().to(label.device)

        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        # 将边的主对角线上的元素置为0
        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2)).unsqueeze(0).unsqueeze(0).repeat(edge.size(0), 1, 1, 1).to(edge.device)
            edge=edge*diag_mask
        
        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

    # def forward(self, s_data, q_data, s_label=None, q_pred_adj=False):
    def forward(self, data ):
        s_data = data['s_data']
        q_data = data['q_data']

        simi_A = calculate_similarity(s_data, q_data)
        simi_A = torch.tensor(simi_A, dtype=torch.float).to(self.device)
  
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb, _ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)

        s_super_emb, q_super_emb, all_super_emb = self.encode_projection(s_emb,q_emb)

        num_supports = int(s_super_emb.size(1))
        num_queries = int(q_super_emb.size(1))

        # TODO init the node edge, delete batch

        node_feature_gd = initialize_nodes_edges(
                                                s_data,
                                                q_data,
                                                num_supports,
                                                num_queries,
                                                2,
                                                self.device
                                                )
        logits_dict, point_similarity, dis_similarity  = self.relationnet(all_super_emb,
                                                                node_feature_gd,
                                                                simi_A
                                                                )
         
        return logits_dict, point_similarity, dis_similarity

    def forward_query_loader(self, s_data, q_data):
        simi_A = calculate_similarity(s_data, q_data)
        simi_A = torch.tensor(simi_A, dtype=torch.float).to(self.device)

        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        y_true_list=[]
   
        # for q_data in q_loader:       
        q_data = q_data.to(s_emb.device)
        y_true_list.append(q_data.y)
        q_emb,_ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
        s_super_emb, q_super_emb, all_super_emb = self.encode_projection(s_emb,q_emb)

        num_supports = int(s_super_emb.size(1))
        num_queries = int(q_super_emb.size(1))

        node_feature_gd = initialize_nodes_edges(
                                                s_data,
                                                q_data,
                                                num_supports,
                                                num_queries,
                                                2,
                                                self.device
                                                )
        logits_dict, point_similarity , dis_similarity= self.relationnet(all_super_emb,
                                                                node_feature_gd,
                                                                simi_A
                                                                )
        return logits_dict,  point_similarity, dis_similarity
       