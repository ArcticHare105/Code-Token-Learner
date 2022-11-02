import sys
import inspect
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch_geometric.nn import GCNConv
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec
special_args = ['edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j']
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')


def weights_init_random(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        tmp_mask = torch.zeros(b, self.heads, n, n, device=x.device, requires_grad=False)
        index = torch.topk(dots, k=int(max(int(n//3), 1)), dim=-1, largest=True)[1]
        tmp_mask.scatter_(-1, index, 1.)
        attn = torch.where(tmp_mask>0, dots, torch.full_like(dots, float('-inf')))

        attn = self.attend(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class InterAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x, y):
        b, nx, _ = x.size()
        b, ny, _ = y.size()        
        h = self.heads

        # q:y kv:x
        q_y = self.to_q(y)
        kv_x = self.to_kv(x).chunk(2, dim = -1)
        qkv_y = (q_y,) + kv_x
        q_y, k_x, v_x = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv_y)

        dots_yx = torch.einsum('b h i d, b h j d -> b h i j', q_y, k_x) * self.scale

        tmp_mask_yx = torch.zeros(b, self.heads, ny, nx, device=x.device, requires_grad=False)
        index = torch.topk(dots_yx, k=int(max(int(nx//4), 1)), dim=-1, largest=True)[1]
        tmp_mask_yx.scatter_(-1, index, 1.)
        attn_yx = torch.where(tmp_mask_yx>0, dots_yx, torch.full_like(dots_yx, float('-inf')))

        attn_yx = self.attend(attn_yx)

        out_y = torch.einsum('b h i j, b h j d -> b h i d', attn_yx, v_x)
        out_y = rearrange(out_y, 'b h n d -> b n (h d)')

        # q:x kv:y
        q_x = self.to_q(x)
        kv_y = self.to_kv(y).chunk(2, dim = -1)
        qkv_x = (q_x,) + kv_y
        q_x, k_y, v_y = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv_x)

        dots_xy = torch.einsum('b h i d, b h j d -> b h i j', q_x, k_y) * self.scale

        tmp_mask_xy = torch.zeros(b, self.heads, nx, ny, device=y.device, requires_grad=False)
        index = torch.topk(dots_xy, k=int(max(int(ny//4), 1)), dim=-1, largest=True)[1]
        tmp_mask_xy.scatter_(-1, index, 1.)
        attn_xy = torch.where(tmp_mask_xy>0, dots_xy, torch.full_like(dots_xy, float('-inf')))

        attn_xy = self.attend(attn_xy)

        out_x = torch.einsum('b h i j, b h j d -> b h i d', attn_xy, v_y)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x), self.to_out(out_y)


class InterTransformer(torch.nn.Module):
    def __init__(self, num_inp, depth, heads, dim_head, num_out):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(num_inp),
                InterAttention(num_inp, heads = heads, dim_head = dim_head),
                PreNorm(num_inp, FeedForward(num_inp, num_out)),
            ]))

    def forward(self, x, y):
        for norm, attn, ff in self.layers:
            x = norm(x)
            y = norm(y)
            # attention
            out_x, out_y = attn(x, y)
            # feed forward
            x = ff(out_x) + x
            y = ff(out_y) + y
        return x, y


class SelfTransformer(torch.nn.Module):
    def __init__(self, num_inp, depth, heads, dim_head, num_out):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_inp, SelfAttention(num_inp, heads = heads, dim_head = dim_head)),
                PreNorm(num_inp, FeedForward(num_inp, num_out)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # attention
            x = attn(x) + x
            # feed forward
            x = ff(x) + x
        return x

# class GCNNet(torch.nn.Module):
#     def __init__(self, vocablen, embedding_dim, out_dim):
#         super().__init__()
#         self.embed = nn.Embedding(vocablen, embedding_dim)
#         self.gcn1 = GCNConv(embedding_dim, embedding_dim)
#         self.gcn2 = GCNConv(embedding_dim, out_dim)        

#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, data):
#         x, edge_index, _ = data
#         x = self.relu(self.gcn1(x, edge_index))
#         att = self.sigmoid(self.gcn2(x, edge_index))
#         return x

# class TransGCNLayer(torch.nn.Module):
#     def __init__(self, num_inp, num_out, heads, dim_head, dropout):
#         super().__init__()
#         self.norm_inp = nn.LayerNorm(num_inp)
#         self.norm_out = nn.LayerNorm(num_inp)        

#         self.gcnlayer = GCNConv(num_inp, num_out)
#         self.attention = Attention(num_inp, heads = heads, dim_head = dim_head, dropout = dropout)

#         self.fead_forward = FeedForward(num_inp, num_out, dropout = dropout)

#     def forward(self, x, edge_index):
#         x_norm = self.norm_inp(x)
#         gcn_embed = self.gcnlayer(x_norm, edge_index)
#         trans_embed = self.attention(x_norm)
#         fusion_embed = x + gcn_embed + trans_embed

#         fusion_norm = self.norm_out(fusion_embed)
#         out = self.fead_forward(fusion_norm)

#         return out + fusion_embed


class FeatureEmbedding(torch.nn.Module):
    def __init__(self, vocablen, num_inp):
        super().__init__()
        self.embed = nn.Embedding(vocablen, num_inp)
        self.code_token = nn.Parameter(torch.randn(1, num_inp))
        torch_init.xavier_uniform_(self.code_token)
        # self.pos_embedding = nn.Parameter(torch.randn(1, 500, num_inp))
        # torch_init.xavier_uniform_(self.pos_embedding)

        self.embedding = GCNConv(num_inp, 8)

    def forward(self, x_input, PE):

        x, edge_index, _, _ = x_input
        encodes = self.embed(x).permute(1,0,2)
        n, t, d = encodes.size()
        att = self.embedding(encodes, edge_index).permute(0,2,1)
        att = F.softmax(att, -1)
        
        encodes = encodes + PE
        learned_tokens = torch.einsum('nkt,ntd->nkd', [att, encodes])

        code_tokens = repeat(self.code_token[None, ...], '() t d -> n t d', n = n)
        x = torch.cat((code_tokens, learned_tokens), dim=1)
        return x, att

class CloneTrans(torch.nn.Module):
    def __init__(self, vocablen, num_inp, heads, dim_head, dropout, num_layers, device):
        super().__init__()
        self.device = device

        self.embedding = FeatureEmbedding(vocablen, num_inp)

        self.intra_transformer = SelfTransformer(num_inp, 3, heads,\
                                                 dim_head, num_inp)

        self.inter_transformer = InterTransformer(num_inp, 1, heads,\
                                                  dim_head, num_inp)

        self.mlp = nn.Linear(num_inp, num_inp)

        channels = int(num_inp / 2)
        self.channels = channels
        df_pe = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("df_pe", df_pe)
        bf_pe = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("bf_pe", bf_pe)

    def get_depth_first_PE(self, df):
        sin_inp_x = torch.einsum("i,j->ij", df, self.df_pe)
        emb = get_emb(sin_inp_x)
        emb_df = torch.zeros((df.shape[0], self.channels), device=df.device).type(df.type())
        emb_df[:, :self.channels] = emb
        return emb_df

    def get_breadth_first_PE(self, bf):
        sin_inp_x = torch.einsum("i,j->ij", bf, self.bf_pe)
        emb = get_emb(sin_inp_x)
        emb_bf = torch.zeros((bf.shape[0], self.channels), device=bf.device).type(bf.type())
        emb_bf[:, :self.channels] = emb
        return emb_bf

    def forward(self, x_input, y_input):
        # import pdb; pdb.set_trace()

        df_x, bf_x = x_input[2], x_input[3]
        df_y, bf_y = y_input[2], y_input[3]

        PE_df_x = self.get_depth_first_PE(df_x)
        PE_df_y = self.get_depth_first_PE(df_y)
        PE_bf_x = self.get_breadth_first_PE(bf_x)
        PE_bf_y = self.get_breadth_first_PE(bf_y)        

        PE_x = torch.cat([PE_df_x, PE_bf_x], -1)
        PE_y = torch.cat([PE_df_y, PE_bf_y], -1)

        x_embed, x_att = self.embedding(x_input, PE_x)
        y_embed, y_att = self.embedding(y_input, PE_y)

        # import pdb; pdb.set_trace()

        x_trans_out = self.intra_transformer(x_embed)
        y_trans_out = self.intra_transformer(y_embed)
        x_out, y_out = self.inter_transformer(x_trans_out, y_trans_out)

        x_out = self.mlp(x_out[:, 0])
        y_out = self.mlp(y_out[:, 0])

        return x_out, y_out, x_att, y_att


