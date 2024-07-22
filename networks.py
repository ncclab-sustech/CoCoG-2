import os
import sys

sys.path.append('../mise')

import torch
from torch import nn
import torch.nn.functional as F
import torch.linalg as LA
    

class SimpleRegEmb(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.proj = nn.Sequential(
            nn.Linear(self.d_in, self.d_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # x(B, d_in)
        y = self.proj(x)
        return y

class SimpleRegMulti(nn.Module):

    def __init__(self, d_in, d_out, l1=0, l2=0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.l1 = l1
        self.l2 = l2
        self.proj = nn.Sequential(
            nn.Linear(self.d_in, self.d_out),
            nn.GELU()
        )

    def forward(self, x1, x2):
        # x1, x2 (B, d_in)
        y = self.proj(x)
        return y
    
    
    # def loss_func(self, x1, x2, m1, m2, return_acc=True, return_loss=True):
    #     y1 = self.proj(x1)
    #     y2 = self.proj(x2)
        
    #     def pairwise_distance_matrix(x):
    #         # Compute pairwise distance matrix.
    #         r = torch.sum(x*x, 1).reshape(-1, 1)
    #         d = r - 2*torch.mm(x, x.t()) + r.t()
    #         d = torch.sqrt(torch.clamp(d, min=0))
    #         return d
        
    #     def accuracy(d, m):
    #         # Nearest neighbor accuracy
    #         nearest = torch.argmin(d, axis=1)
    #         acc = (nearest == torch.argmin(m, axis=1)).float().mean()
    #         return acc

    #     # Calculate distance matrices
    #     d1 = pairwise_distance_matrix(y1)
    #     d2 = pairwise_distance_matrix(y2)

    #     output = []

    #     if return_acc:
    #         acc1 = accuracy(d1, m1)
    #         acc2 = accuracy(d2, m2)
    #         output.append((acc1 + acc2) / 2)
        
    #     if return_loss:
    #         loss = F.mse_loss(d1, m1) + F.mse_loss(d2, m2)
    #         output.append(loss)

    #     return output

class SimpleRegProb(nn.Module):

    def __init__(self, d_in, d_out, l1=0, l2=0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.proj = nn.Sequential(
            nn.Linear(self.d_in, self.d_out),
            nn.LeakyReLU()
        )
        self.l1 = l1
        self.l2 = l2

    def forward(self, x):
        # x(B, d_in)
        y = self.proj(x)
        return y

    def similarity_log_gaussian_kernel(self, x1, x2):
        return - LA.norm(x1 - x2, dim=-1)

    def cal_sim(self, h):
        # h (N, 3, l)
        s12 = (h[:,0] * h[:,1]).sum(dim=-1) # (N, )
        s13 = (h[:,0] * h[:,2]).sum(dim=-1)
        s23 = (h[:,1] * h[:,2]).sum(dim=-1)
        # s12 = self.similarity_log_gaussian_kernel(h[:,0], h[:,1]) # (N, )
        # s13 = self.similarity_log_gaussian_kernel(h[:,0], h[:,2])
        # s23 = self.similarity_log_gaussian_kernel(h[:,1], h[:,2])

        S = torch.stack((s23, s13, s12), dim=1) # (N, 3)
        return S, S.argmax(dim=-1)==2
    
    def l_CE(self, S):
        # maximize the option 3 (s12)
        target = 2 * torch.ones(S.shape[0], device=S.device, dtype=torch.int64)
        l_ce = F.cross_entropy(S, target)
        return l_ce

    def l2_loss(self):
        l2_loss = self.proj[0].weight.square().mean()
        return self.l2 * l2_loss
    
    def loss_func(self, h, tr, return_acc=True, return_loss=True):
        # tr (B, 3)
        # h (N, d_in)
        x = tr @ h # (B, 3, d_in)
        B, _, _ = x.shape
        y = self(x.view(-1, self.d_in)).view(B, 3, self.d_out) # (B, 3, d_out)
        output = []
        output.append(y)

        l1_loss = self.l1 * y.abs().mean()
        if return_acc:
            S, acc = self.cal_sim(y)
            output.append(acc)
        if return_loss:
            l = self.l_CE(S)
            l += self.l2_loss() + l1_loss
            output.append(l)
        return output

class MLPProb(nn.Module):

    def __init__(self, d_in, d_out, d_hidden=1024, hidden_layer=3, l1=0, l2=0, dropout=0.1, final_act=nn.Softplus):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.input_layer = nn.Sequential(
            nn.Linear(self.d_in, self.d_hidden),
            nn.LayerNorm(self.d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        if hidden_layer is not None:
            self.hidden_layer = nn.Sequential(
                *[nn.Sequential(
                    nn.Linear(self.d_hidden, self.d_hidden), 
                    nn.LayerNorm(self.d_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ) for _ in range(hidden_layer)]
            )
        else: self.hidden_layer = None
        self.proj = nn.Sequential(
            nn.Linear(self.d_hidden, self.d_out),
            final_act()
        )
        self.l1 = l1
        self.l2 = l2

    def forward(self, x):
        # x(B, d_in)
        x = self.input_layer(x)
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
        y = self.proj(x)
        return y

    def similarity_log_gaussian_kernel(self, x1, x2):
        return - LA.norm(x1 - x2, dim=-1)

    def cal_sim(self, h):
        # h (N, 3, l)
        s12 = (h[:,0] * h[:,1]).sum(dim=-1) # (N, )
        s13 = (h[:,0] * h[:,2]).sum(dim=-1)
        s23 = (h[:,1] * h[:,2]).sum(dim=-1)
        # s12 = self.similarity_log_gaussian_kernel(h[:,0], h[:,1]) # (N, )
        # s13 = self.similarity_log_gaussian_kernel(h[:,0], h[:,2])
        # s23 = self.similarity_log_gaussian_kernel(h[:,1], h[:,2])

        S = torch.stack((s23, s13, s12), dim=1) # (N, 3)
        return S, S.argmax(dim=-1)==2
    
    def l_CE(self, S):
        # maximize the option 3 (s12)
        target = 2 * torch.ones(S.shape[0], device=S.device, dtype=torch.int64)
        l_ce = F.cross_entropy(S, target)
        return l_ce

    def l2_loss(self):
        l2_loss = 0
        if self.hidden_layer is not None:
            for p in self.hidden_layer.parameters():
                l2_loss += p.square().mean()
        l2_loss += self.proj[0].weight.square().mean()
        return self.l2 * l2_loss
    
    def loss_func(self, h, tr, return_acc=True, return_loss=True):
        # tr (B, 3, N)
        # h (N, d_in) or (N, *)
        x = tr @ h # (B, 3, d_in)
        B, _, _ = x.shape
        y = self(x.view(-1, self.d_in)).view(B, 3, self.d_out) # (B, 3, d_out)
        output = []
        output.append(y)

        l1_loss = self.l1 * y.abs().mean()
        if return_acc:
            S, acc = self.cal_sim(y)
            output.append(acc)
        if return_loss:
            l = self.l_CE(S)
            l += self.l2_loss() + l1_loss
            output.append(l)
        return output