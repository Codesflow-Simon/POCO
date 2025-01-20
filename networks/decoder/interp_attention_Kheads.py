import torch
from lightconvpoint.spatial import knn
from lightconvpoint.utils.functional import batch_gather
from torch_geometric.data import Data
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

class InterpAttentionKHeadsNet(torch.nn.Module):

    def __init__(self, latent_size, out_channels, K=16):
        super().__init__()
        
        logging.info(f"InterpNet - Simple - K={K}")
        self.fc1 = torch.nn.Conv2d(latent_size+3, latent_size, 1)
        self.bn1 = torch.nn.BatchNorm2d(latent_size)
        
        self.fc2 = torch.nn.Conv2d(latent_size, latent_size, 1)
        self.bn2 = torch.nn.BatchNorm2d(latent_size)
        
        self.fc3 = torch.nn.Conv2d(latent_size, latent_size, 1)
        self.bn3 = torch.nn.BatchNorm2d(latent_size)

        self.fc_query = torch.nn.Conv2d(latent_size, 64, 1)
        self.bn_query = torch.nn.BatchNorm2d(64)
        
        self.fc_value = torch.nn.Conv2d(latent_size, latent_size, 1)
        self.bn_value = torch.nn.BatchNorm2d(latent_size)

        self.fc8 = torch.nn.Conv1d(latent_size, out_channels, 1)
        self.activation = torch.nn.ReLU()

        self.k = K
        self.iter_count = 0
        self.save_dir = None

    def set_save_dir(self, save_dir):
        """Set the directory where visualizations will be saved"""
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)

    def forward_spatial(self, data):

        pos = data["pos"]
        pos_non_manifold = data["pos_non_manifold"]

        add_batch_dimension_pos = False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            add_batch_dimension_pos = True

        add_batch_dimension_non_manifold = False
        if len(pos_non_manifold.shape) == 2:
            pos_non_manifold = pos_non_manifold.unsqueeze(0)
            add_batch_dimension_non_manifold = True

        if pos.shape[1] != 3:
            pos = pos.transpose(1,2)

        if pos_non_manifold.shape[1] != 3:
            pos_non_manifold = pos_non_manifold.transpose(1,2)

        indices = knn(pos, pos_non_manifold, self.k)

        if add_batch_dimension_non_manifold or add_batch_dimension_pos:
            indices = indices.squeeze(0)

        ret_data = {}
        ret_data["proj_indices"] = indices

        return ret_data

    def visualize_knn(self, pos, pos_non_manifold, indices, sample_idx=0, point_idx=0):
        """Visualize KNN neighbors for a specific query point"""
        if self.save_dir is None:
            return

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get positions in correct format
        pos = pos[sample_idx].transpose(0,1).detach().cpu().numpy()  # [N, 3]
        query = pos_non_manifold[sample_idx].transpose(0,1).detach().cpu().numpy()  # [M, 3]
        neighbors = pos[indices[sample_idx, point_idx].detach().cpu().numpy()]  # [K, 3]

        # Plot all points
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='blue', alpha=0.1, label='Point Cloud')
        
        # Plot query point
        query_point = query[point_idx]
        ax.scatter(query_point[0], query_point[1], query_point[2], 
                  c='red', s=100, label='Query Point')
        
        # Plot neighbors
        ax.scatter(neighbors[:,0], neighbors[:,1], neighbors[:,2], 
                  c='green', s=50, label='KNN neighbors')
        
        # Draw lines from query to neighbors
        for neighbor in neighbors:
            ax.plot([query_point[0], neighbor[0]], 
                   [query_point[1], neighbor[1]], 
                   [query_point[2], neighbor[2]], 'g--', alpha=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title(f'KNN visualization (K={self.k}) - Iteration {self.iter_count}')
        plt.savefig(os.path.join(self.save_dir, 'visualizations', f'knn_iter_{self.iter_count}.png'))
        plt.close()

    def visualize_attention_weights(self, attention, sample_idx=0, point_idx=0):
        """Visualize attention weights for a specific query point"""
        if self.save_dir is None:
            return

        weights = attention[sample_idx, point_idx].detach().cpu().numpy()
        
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(weights)), weights)
        plt.xlabel('Neighbor Index')
        plt.ylabel('Attention Weight')
        plt.title(f'Attention Weights Distribution - Iteration {self.iter_count}')
        plt.savefig(os.path.join(self.save_dir, 'visualizations', f'attention_weights_iter_{self.iter_count}.png'))
        plt.close()

    def visualize_relative_coords(self, pos_diff, sample_idx=0, point_idx=0):
        """Visualize relative coordinates for a specific query point"""
        if self.save_dir is None:
            return

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get relative coordinates for specific point
        rel_coords = pos_diff[sample_idx, :, point_idx].detach().cpu().numpy()  # [3, K]
        
        # Plot origin (query point)
        ax.scatter(0, 0, 0, c='red', s=100, label='Query Point (Origin)')
        
        # Plot relative positions of neighbors
        ax.scatter(rel_coords[0], rel_coords[1], rel_coords[2], 
                  c='blue', label='Relative Neighbor Positions')
        
        # Draw lines from origin to relative positions
        for i in range(rel_coords.shape[1]):
            ax.plot([0, rel_coords[0,i]], 
                   [0, rel_coords[1,i]], 
                   [0, rel_coords[2,i]], 'b--', alpha=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title(f'Relative Coordinates Visualization - Iteration {self.iter_count}')
        plt.savefig(os.path.join(self.save_dir, 'visualizations', f'relative_coords_iter_{self.iter_count}.png'))
        plt.close()

    def forward(self, data, spatial_only=False, spectral_only=False, last_layer=True, return_last_features=False, visualize=False):
        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:
            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value

        x = data["latents"]
        indices = data["proj_indices"]
        pos = data["pos"]
        pos_non_manifold = data["pos_non_manifold"]

        if pos.shape[1] != 3:
            pos = pos.transpose(1,2)

        if pos_non_manifold.shape[1] != 3:
            pos_non_manifold = pos_non_manifold.transpose(1,2)

        x = batch_gather(x, 2, indices)
        pos = batch_gather(pos, 2, indices)
        pos = pos_non_manifold.unsqueeze(3) - pos

        if visualize and self.save_dir is not None:
            self.visualize_knn(data["pos"], data["pos_non_manifold"], indices)
            self.visualize_relative_coords(pos)

        x = torch.cat([x,pos], dim=1)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))

        query = self.bn_query(self.fc_query(x))
        value = self.bn_value(self.fc_value(x))

        attention = torch.nn.functional.softmax(query, dim=-1).mean(dim=1)
        
        if visualize and self.save_dir is not None:
            self.visualize_attention_weights(attention)
            self.iter_count += 1

        x = torch.matmul(attention.unsqueeze(-2), value.permute(0,2,3,1)).squeeze(-2)
        x = x.transpose(1,2)

        if return_last_features:
            xout = self.fc8(x)
            return xout, x

        if last_layer:
            x = self.fc8(x)

        return x
        

 
