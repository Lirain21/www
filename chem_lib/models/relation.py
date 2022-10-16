
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers,batch_norm=False, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
            if l < num_layers - 1:
                if batch_norm:
                    layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=hidden_dim)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
            in_dim = hidden_dim
        if num_layers > 0:
            self.network = nn.Sequential(layer_list)
        else:
            self.network = nn.Identity()

    def forward(self, emb):
        out = self.network(emb)
        return out

class Attention(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """
    def __init__(self, dim, num_heads=2, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class ContextMLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers,pre_fc=0,batch_norm=False, dropout=0.,ctx_head=2):
        super(ContextMLP, self).__init__()

        self.pre_fc = pre_fc #0, 1
        in_dim = inp_dim
        out_dim = hidden_dim

        if self.pre_fc:
            hidden_dim=int(hidden_dim//2)  
            self.attn_layer = Attention(hidden_dim,num_heads=ctx_head,attention_dropout=dropout)        
            self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)
        else:
            self.attn_layer = Attention(inp_dim, num_heads=ctx_head, attention_dropout=dropout)
            inp_dim=int(inp_dim*2)
            self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)

    def forward(self, s_emb, q_emb):

        n_support = s_emb.size(0)
        n_query = q_emb.size(0)

        n_shot=int(n_support//2)
        # s_emb_neg = s_emb[:n_shot,:].mean(0).unsqueeze(0)
        # s_emb_pos = s_emb[n_shot:,:].mean(0).unsqueeze(0)
        # s_emb = torch.cat((s_emb[:n_shot,:], s_emb_neg, s_emb[n_shot:,:], s_emb_pos), dim=0)
        s_emb_rep = s_emb.unsqueeze(0).repeat(n_query, 1,1)
        q_emb_rep = q_emb.unsqueeze(1)
        all_emb = torch.cat((s_emb_rep, q_emb_rep), 1)
        orig_all_emb = all_emb

        neg_proto_emb_2= all_emb[:,:n_shot].mean(1).unsqueeze(1).repeat(1,n_support+1, 1) # size([16,21,300])
        pos_proto_emb_2 =all_emb[:,n_shot:2*n_shot].mean(1).unsqueeze(1).repeat(1,n_support+1,1) # sie([16, 21, 300])
        
        all_emb_super_1 = torch.stack((all_emb, neg_proto_emb_2, pos_proto_emb_2), -2)

        q,s,n, d = all_emb_super_1.shape
        x=all_emb_super_1.reshape((q*s,n,d))
        attn_x =self.attn_layer(x)
        attn_x=attn_x.reshape((q,s,n, d))
        all_emb_super_1 = attn_x[:,:,0,]

        # all_emb.size([16,23,600])
        all_emb_super_1 = torch.cat([all_emb_super_1, orig_all_emb],dim=-1)

        # all_emb.size([16,23,128])     
        all_emb_super_1 = self.mlp_proj(all_emb_super_1)
        
        s_super_emb = all_emb_super_1[0,:-1,:].squeeze(0)
        q_super_emb = all_emb_super_1[:,-1,:].squeeze(1)
        all_super_emb = torch.cat((s_super_emb, q_super_emb), dim=0)
        s_super_emb = s_super_emb.unsqueeze(0)
        q_super_emb = q_super_emb.unsqueeze(0)
        all_super_emb = all_super_emb.unsqueeze(0)  
        return s_super_emb, q_super_emb, all_emb_super_1


