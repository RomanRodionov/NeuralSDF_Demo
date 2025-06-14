import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import trimesh
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_sdf, get_surface_point_cloud
import os

import open3d as o3d

class SDF_Dataset(Dataset):
    def __init__(self, dataset_size, batch_size, model_path, checkpoint="checkpoints/sdf_data.npy", normalize=True, uniform_ratio=0.1):
        super().__init__()

        self.batch_size = batch_size

        if checkpoint:
            if os.path.isfile(checkpoint):
                data = np.load(checkpoint, allow_pickle=True).item()
            else:
                data = self.generate_points(model_path, dataset_size, normalize, uniform_ratio)
                np.save(checkpoint, data)

        self.points, self.distances = data["points"], data["distances"]

        self.points = torch.tensor(self.points)
        self.distances = torch.tensor(self.distances)
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        return {
            'points': self.points[index],
            'dist': self.distances[index]
        }
    
    def generate_points(self, path, n, normalize=True, uniform_ratio=0.0, uniform_bounds=(-1.2, 1.2), visualize=False):
        # normalize=True => map points to [-1, 1]
        mesh = trimesh.load(path)

        if normalize:
            center = mesh.bounds.mean(axis=0)
            mesh.apply_translation(-center)

            scale = 2.0 / mesh.extents.max()
            mesh.apply_scale(scale)

        #mesh.export("mesh_gt.obj")

        n_uniform = int(n * uniform_ratio)
        n_surface = n - n_uniform

        if uniform_ratio < 1.0:            
            surface_point_cloud = get_surface_point_cloud(mesh)
            surface_points, surface_sdf = surface_point_cloud.sample_sdf_near_surface(number_of_points=n_surface)
        else:
            surface_points = np.random.uniform(-1, 1, size=(0, 3)).astype(np.float32)
            surface_sdf = np.array([])

        if uniform_ratio > 0:
            uniform_points = np.random.uniform(*uniform_bounds, size=(n_uniform, 3)).astype(np.float32)
            uniform_sdf = mesh_to_sdf(mesh, uniform_points).astype(np.float32)
        else:
            uniform_points = np.random.uniform(*uniform_bounds, size=(0, 3)).astype(np.float32)
            uniform_sdf = np.array([])

        if visualize:
            pcd1 = o3d.geometry.PointCloud()
            pcd2 = o3d.geometry.PointCloud()

            pcd1.points = o3d.utility.Vector3dVector(surface_points)
            pcd2.points = o3d.utility.Vector3dVector(uniform_points)

            pcd1.paint_uniform_color([1, 0, 0])
            pcd2.paint_uniform_color([0, 1, 0])

            o3d.visualization.draw_geometries([pcd1, pcd2])

        all_points = np.concatenate([surface_points, uniform_points], axis=0)
        all_sdf = np.concatenate([surface_sdf, uniform_sdf], axis=0)[:, np.newaxis]

        return {
            'points': all_points.astype(np.float32),
            'distances': all_sdf.astype(np.float32)
        }
    
    def n_batches(self):
        return len(self) // self.batch_size

    def get_batch(self, index):
        l = index * self.batch_size
        r = l + self.batch_size
        
        return {
            'points': self.points[l:r],
            'dist': self.distances[l:r]
        }
    
    def cuda(self):
        self.points = self.points.cuda()
        self.distances = self.distances.cuda()
        return self
    
    def shuffle(self):
        ind = torch.randperm(len(self.points))
        self.points = self.points[ind]
        self.distances = self.distances[ind]
        return self