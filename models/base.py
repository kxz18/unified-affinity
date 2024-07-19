#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_sum

from data.format import VOCAB
from .GET.tools import knn_edges, fully_connect_edges


ReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 'noise', 'noise_level',
     'unit_repr', 'block_repr', 'graph_repr',
     'batch_id', 'block_id',
     'loss', 'noise_loss', 'noise_level_loss', 'align_loss'],
    )


# embedding of blocks (for proteins, it is residue).
class BlockEmbedding(nn.Module):
    '''
    [atom embedding + block embedding + atom position embedding]
    '''
    def __init__(self, num_block_type, num_atom_type, num_atom_position, embed_size, no_block_embedding=False):
        super().__init__()
        if not no_block_embedding:
            self.block_embedding = nn.Embedding(num_block_type, embed_size)
        self.no_block_embedding = no_block_embedding
        self.atom_embedding = nn.Embedding(num_atom_type, embed_size)
        self.position_embedding = nn.Embedding(num_atom_position, embed_size)
    
    def forward(self, B, A, atom_positions, block_id):
        '''
        :param B: [Nb], block (residue) types
        :param A: [Nu], unit (atom) types
        :param atom_positions: [Nu], unit (atom) position encoding
        :param block_id: [Nu], block id of each unit
        '''
        atom_embed = self.atom_embedding(A) + self.position_embedding(atom_positions)
        if self.no_block_embedding:
            return atom_embed
        block_embed = self.block_embedding(B[block_id])
        return atom_embed + block_embed
    

class DenoisePretrainModel(nn.Module):

    def __init__(self, model_type, hidden_size, n_channel,
                 n_rbf=1, cutoff=7.0, n_head=1,
                 radial_size=16, edge_size=64, k_neighbors=9, n_layers=3,
                 sigma_begin=10, sigma_end=0.01, n_noise_level=50,
                 dropout=0.1, std=10,
                 no_block_embedding=False) -> None:
        super().__init__()

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_channel = n_channel
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.n_head = n_head
        self.radial_size = radial_size
        self.edge_size = edge_size
        self.k_neighbors = k_neighbors
        self.n_layers = n_layers
        self.dropout = dropout
        self.std = std
        self.no_block_embedding = no_block_embedding

        self.block_embedding = BlockEmbedding(
            num_block_type=len(VOCAB),
            num_atom_type=VOCAB.get_num_atom_type(),
            num_atom_position=VOCAB.get_num_atom_pos(),
            embed_size=hidden_size,
            no_block_embedding=no_block_embedding
        )

        self.edge_embedding = nn.Embedding(2, edge_size)  # [intra / inter]
        
        z_requires_grad = False
        if model_type == 'GET':
            from .GET.encoder import GETEncoder
            self.encoder = GETEncoder(
                hidden_size, radial_size, n_channel,
                n_rbf, cutoff, edge_size, n_layers,
                n_head, dropout=dropout,
                z_requires_grad=z_requires_grad
            )
        else:
            raise NotImplementedError(f'Model type {model_type} not implemented!')
        
        self.energy_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        # TODO: add zero noise level
        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), n_noise_level)), dtype=torch.float)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # [n_noise_level]

    @torch.no_grad()
    def choose_receptor(self, batch_size, device):
        segment_retain = (torch.randn((batch_size, ), device=device) > 0).long()  # [bs], 0 or 1
        return segment_retain

    @torch.no_grad()
    def normalize(self, Z, B, block_id, batch_id, segment_ids, receptor_segment):
        return Z
        # centering
        center = Z[(B[block_id] == self.global_block_id) & (segment_ids[block_id] == receptor_segment[batch_id][block_id])]  # [bs]
        Z = Z - center[batch_id][block_id]
        # normalize
        Z = Z / self.std
        return Z

    @torch.no_grad()
    def perturb(self, Z, block_id, batch_id, batch_size, segment_ids, receptor_segment):

        noise_level = torch.randint(0, self.sigmas.shape[0], (batch_size,), device=Z.device)
        # noise_level = torch.ones((batch_size, ), device=Z.device, dtype=torch.long) * (self.sigmas.shape[0] - 1)
        used_sigmas = self.sigmas[noise_level][batch_id]  # [Nb]
        used_sigmas = used_sigmas[block_id]  # [Nu]

        # # randomly select one side to perturb (segment type 0 or segment type 1)
        # perturb_block_mask = segment_ids == receptor_segment[batch_id]  # [Nb]
        # perturb_mask = perturb_block_mask[block_id]  # [Nu]
        perturb_mask = torch.ones_like(segment_ids, dtype=torch.bool)

        # used_sigmas[~perturb_mask] = 0  # only one side of the complex is perturbed

        noise = torch.randn_like(Z)  # [Nu, channel, 3]

        Z_perturbed = Z + noise * used_sigmas.unsqueeze(-1).unsqueeze(-1)

        return Z_perturbed, noise, noise_level, perturb_mask
    
    # def pred_noise_from_energy(self, energy, Z):
    #     grad_outputs = [torch.ones_like(energy)]
    #     dy = grad(
    #         [energy],
    #         [Z],
    #         grad_outputs=grad_outputs,
    #         create_graph=self.training,
    #         retain_graph=self.training,
    #     )[0]
    #     pred_noise = (-dy).view(-1, self.n_channel, 3).contiguous() # the direction of the gradients is where the energy drops the fastest. Noise adopts the opposite direction
    #     return pred_noise

    def get_edges(self, B, batch_id, segment_ids, Z, block_id):
        row, col = fully_connect_edges(batch_id)
        is_intra = segment_ids[row] == segment_ids[col]
        intra_edges = torch.stack([row[is_intra], col[is_intra]], dim=0)
        inter_edges = torch.stack([row[~is_intra], col[~is_intra]], dim=0)
        intra_edges = knn_edges(block_id, batch_id, Z, self.k_neighbors, intra_edges)
        inter_edges = knn_edges(block_id, batch_id, Z, self.k_neighbors, inter_edges)
        
        edges = torch.cat([intra_edges, inter_edges], dim=1)
        edge_attr = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        edge_attr = self.edge_embedding(edge_attr)

        return edges, edge_attr

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=True, return_loss=True) -> ReturnValue:

        # batch_id and block_id
        with torch.no_grad():

            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            batch_size = lengths.shape[0]
            # select receptor
            receptor_segment = self.choose_receptor(batch_size, batch_id.device)
            # normalize
            Z = self.normalize(Z, B, block_id, batch_id, segment_ids, receptor_segment)
            # perturbation
            Z_perturbed, noise, noise_level, perturb_mask = self.perturb(Z, block_id, batch_id, batch_size, segment_ids, receptor_segment)

        #Z_perturbed.requires_grad_(True)

        H_0 = self.block_embedding(B, A, atom_positions, block_id)

        # encoding
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, Z_perturbed, block_id)
        unit_repr, block_repr, graph_repr, pred_Z = self.encoder(H_0, Z_perturbed, block_id, batch_id, edges, edge_attr)

        # predict energy
        # must be sum instead of mean! mean will make the gradient (predicted noise) pretty small, and the score net will easily converge to 0
        pred_energy = scatter_sum(self.energy_ffn(block_repr).squeeze(-1), batch_id)

        # predict noise level
        # pred_noise_level = self.noise_level_ffn(graph_repr)  # [batch_size, n_noise_level]

        if return_noise or return_loss:
            # predict noise
            pred_noise = pred_Z - Z_perturbed
            #pred_noise = self.pred_noise_from_energy(pred_energy, Z_perturbed)
        else:
            pred_noise = None

        if return_loss:
            # noise loss
            noise_loss = F.mse_loss(pred_noise[perturb_mask], noise[perturb_mask], reduction='none')  # [Nperturb, n_channel, 3]
            noise_loss = noise_loss.sum(dim=-1).sum(dim=-1)  # [Nperturb]
            noise_loss = scatter_sum(noise_loss, batch_id[block_id][perturb_mask])  # [batch_size]
            noise_loss = 0.5 * noise_loss.mean()  # [1]

            align_loss = 0

            noise_level_loss = 0

            # total loss
            loss = noise_loss # + noise_level_loss # + align_loss

        else:
            noise_loss, align_loss, noise_level_loss, loss = None, None, None, None

        return ReturnValue(

            # denoising variables
            energy=pred_energy,
            noise=pred_noise,
            noise_level=0,
            # noise_level=torch.argmax(pred_noise_level, dim=-1),

            # representations
            unit_repr=unit_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            loss=loss,
            noise_loss=noise_loss,
            noise_level_loss=noise_level_loss,
            align_loss=align_loss
        )