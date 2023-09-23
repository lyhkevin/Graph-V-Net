import math
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import numpy as np
import torch
from torch import nn
from models.gcn_lib.torch_nn import BasicConv, batched_index_select, act_layer
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--channels', default=384)
        self.parser.add_argument('--num_class', default=384)
        self.opt = self.parser.parse_args(args=[])
    def get_opt(self):
        return self.opt

class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, k, act='relu', norm=None, bias=True):
        super(GraphAttention, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)
        self.graph_attention = graph_attention(dim=in_channels, embed_dim=in_channels,k=k)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = self.graph_attention(x,x_j)
        temp = x
        x = self.nn(torch.cat([x, x_j], dim=1))
        x = self.relu(x)
        x = x + temp
        return x

class graph_attention(nn.Module):
    def __init__(self, dim, embed_dim,k):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.k = k
        self.embedding = nn.Linear(dim,embed_dim)
        self.attention = nn.Linear(2 * embed_dim, 1)
        self.relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, x_n):
        x = x.repeat(1,1,1,self.k)
        x = rearrange(x,'b d x y -> b x y d')
        x_n = rearrange(x_n, 'b d x y -> b x y d')
        e, e_n = self.embedding(x), self.embedding(x_n)
        e = torch.concat((e, e_n), dim=3)
        attn = self.relu(self.attention(e))
        attn = self.softmax(attn)
        x_n = x_n * attn
        x_n = torch.sum(x_n, dim=2)
        x_n = rearrange(x_n, 'b x y -> b y x').unsqueeze(dim=-1)
        return x_n

def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

class knn(torch.nn.Module):
    def __init__(self,k):
        super(knn, self).__init__()
        self.k = k

    def forward(self, x):
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        x = F.normalize(x, p=2.0, dim=1)
        dist = pairwise_distance(x.detach())
        _, nn_idx = torch.topk(-dist, k=self.k)  #b, n, k
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1)
        edge_index = torch.stack((nn_idx, center_idx), dim=0)
        return edge_index

class knn_euclidean(torch.nn.Module):

    def __init__(self,k):
        super(knn_euclidean, self).__init__()
        self.k = k
        x = torch.arange(8)
        y = torch.arange(8)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coord = torch.stack([grid_x,grid_y],dim=0).to(torch.float32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.coord = rearrange(coord, 'd h w -> d (h w)').to(device)

    def forward(self, x):
        batch = x.size()[0]
        x = self.coord.repeat(batch, 1, 1)
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        x = F.normalize(x, p=2.0, dim=1)
        dist = pairwise_distance(x.detach())
        _, nn_idx = torch.topk(-dist, k=self.k)  #b, n, k
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1)
        edge_index = torch.stack((nn_idx, center_idx), dim=0)
        return edge_index

class Downsample(nn.Module):

    def __init__(self, in_dim=384, out_dim=384):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, Upsample=True,bilinear=True):
        super().__init__()
        self.Upsample = Upsample
        if self.Upsample == True:
            if bilinear == 'bilinear':
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        if self.Upsample == True:
            x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Down_block(torch.nn.Module):
    def __init__(self,in_channels, out_channels, k, euclidean=False):
        super(Down_block, self).__init__()
        self.k = k
        if euclidean == False:
            self.knn_graph = knn(k=self.k)
        else:
            self.knn_graph = knn_euclidean(k=self.k)
        self.conv = GraphAttention(in_channels, in_channels, k=self.k, act='relu', norm='batch', bias=True)
        self.down = Downsample(in_channels, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.knn_graph(x)
        x = self.conv(x, edge_index)
        x = x.reshape(B, -1, H, W).contiguous()
        x = self.down(x)
        return x

class Up_block(torch.nn.Module):

    def __init__(self,in_channels, out_channels, k):
        super(Up_block, self).__init__()
        self.k = k
        self.knn_graph = knn(k=self.k)
        self.conv = GraphAttention(in_channels, in_channels, k=self.k, act='relu', norm='batch', bias=True)
        self.up = Up(in_channels, out_channels)

    def forward(self, x1, x2):
        x = self.up(x1, x2)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.knn_graph(x)
        x = self.conv(x, edge_index)
        x = x.reshape(B, -1, H, W).contiguous()
        return x

class Graph_VNet(torch.nn.Module):
    def __init__(self, opt):
        super(Graph_VNet, self).__init__()
        self.channels = opt.channels
        self.num_class = opt.num_class
        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels, 8, 8))
        self.down1 = Down_block(in_channels = self.channels, out_channels = self.channels, k=16, euclidean=True)
        self.down2 = Down_block(in_channels = self.channels, out_channels = self.channels, k=5)
        self.down3 = Down_block(in_channels = self.channels, out_channels = self.channels, k=3)
        self.up1 = Up_block(in_channels = self.channels, out_channels = self.channels, k=3)
        self.up2 = Up_block(in_channels = self.channels, out_channels = self.channels, k=5)
        self.up3 = Up_block(in_channels = self.channels, out_channels=self.channels, k=10)
        self.model_init()

        self.pred_1 = Seq(nn.Conv2d(self.channels, 1024, 1, bias=True),
                        nn.BatchNorm2d(1024), act_layer('gelu'),
                        nn.Conv2d(1024, self.num_class, 1, bias=True))

        self.pred_2 = Seq(nn.Conv2d(self.channels, 1024, 1, bias=True),
                        nn.BatchNorm2d(1024),act_layer('gelu'),
                        nn.Conv2d(1024, 4, 1, bias=True))

        self.pred_4 = Seq(nn.Conv2d(self.channels, 1024, 1, bias=True),
                        nn.BatchNorm2d(1024),act_layer('gelu'),
                        nn.Conv2d(1024, 4, 1, bias=True))

        self.pred_8 = Seq(nn.Conv2d(self.channels, 1024, 1, bias=True),
                          nn.BatchNorm2d(1024),act_layer('gelu'),
                          nn.Conv2d(1024, 4, 1, bias=True))

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = x + self.pos_embed # batch_size, dim, 8, 8
        tmp = x
        x1 = self.down1(x) # batch_size, dim, 4, 4
        x2 = self.down2(x1) # batch_size, dim, 2, 2
        x3 = self.down3(x2) # batch_size, dim, 1, 1
        pred_1 = self.pred_1(x3)
        x = self.up1(x3,x2) # batch_size, dim, 2, 2
        pred_2 = self.pred_2(x)
        x = self.up2(x,x1) # batch_size, dim, 4, 4
        pred_4 = self.pred_4(x)
        x = self.up3(x,tmp) # batch_size, dim, 8, 8
        pred_8 = self.pred_8(x)
        pred_1 = rearrange(pred_1, 'b d h w -> (b h w) d')
        pred_2 = rearrange(pred_2, 'b d h w -> (b h w) d')
        pred_4 = rearrange(pred_4, 'b d h w -> (b h w) d')
        pred_8 = rearrange(pred_8, 'b d h w -> (b h w) d')
        return pred_1, pred_2, pred_4, pred_8

if __name__ == '__main__':
    opt = Options().get_opt()
    model = Graph_VNet(opt)
    input = torch.zeros((2,384,8,8))
    pred_1, pred_2, pred_4, pred_8 = model(input)
    print(pred_1.shape, pred_2.shape, pred_4.shape, pred_8.shape)
