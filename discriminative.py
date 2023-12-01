import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminative(nn.Module):
    
    def __init__(self, alpha, k):
        super(Discriminative, self).__init__()
        self.alpha = alpha
        self.k = k
              
    def forward(self, original, hidden):
        batch_size = original.size(0)
        #print(f'batch: {batch_size} original: {original.shape} hidden:{hidden.shape}')
        p = original.unsqueeze(1)
        q = original.unsqueeze(0)
        distance_matrix = torch.sqrt(torch.sum((p - q) ** 2, dim=2))
        #print(f'p:{p.shape} q:{q.shape} dis:{distance_matrix.shape}')

        # anchor pairs and non-anchor pairs
        idx = torch.argsort(distance_matrix, dim=1)
        ranks = torch.argsort(idx, dim=1)
        # #print(f'\n ranks: {ranks.shape} \n idx: {idx.shape}')
        anchor_matrix = (ranks < self.k).float()
        #print(f'\n anchor_matrix: {anchor_matrix.shape} \n')

        
        
        # Symmetrize anchor matrix
        anchor_matrix = anchor_matrix + anchor_matrix.transpose(0, 1)
        anchor_matrix = torch.minimum(anchor_matrix, torch.ones_like(anchor_matrix))
        #print(f'\n anchor_matrix final: {anchor_matrix.shape} \n')
        device = anchor_matrix.get_device()
        
        # non-anchor matrix
        matrix_of_1s = torch.ones(batch_size, batch_size, dtype=torch.float32).to(device)
        non_anchor_matrix = matrix_of_1s - anchor_matrix

        #print(f'non_anchor_matrix:{non_anchor_matrix.shape}  anchor_matrix:{anchor_matrix.shape}')
        #print(f'\n hidden:{hidden.shape} \n ')
        # L1 norm
        similarity = F.normalize(hidden, dim=1)
        #print(f'\n  sim: {similarity.shape}')

        C = torch.matmul(similarity, similarity.transpose(0, 1))

        #print(f'C: {C.shape}')

        # normalization factor
        Nw = (1 - self.alpha) / torch.sum(anchor_matrix)
        Nb = 1 / (torch.sum(non_anchor_matrix) - torch.sum(anchor_matrix))
        #print(f'Nw: {Nw} Nb:{Nb}')
        # final calculation for between cluster and within cluster
        non_anchor = Nb * torch.mul(C, non_anchor_matrix)
        anchor = Nw * torch.mul(C, anchor_matrix)
        
        
        return non_anchor, anchor
