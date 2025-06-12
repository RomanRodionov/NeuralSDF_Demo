import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
import os

class SDF_Dataset(Dataset):
    def __init__(self, dataset_size, batch_size, model_path, checkpoint="checkpoints/sdf_data.npy", normalize=True):
        super().__init__()

        self.batch_size = batch_size

        if checkpoint:
            if os.path.isfile(checkpoint):
                data = np.load(checkpoint, allow_pickle=True).item()
            else:
                data = self.generate_points(model_path, dataset_size, normalize)
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
    
    def generate_points(self, path, n, normalize=True):
        # normalize=True => map points to [-1, 1]
        mesh = trimesh.load(path)

        if normalize:
            center = mesh.bounds.mean(axis=0)
            mesh.apply_translation(-center)

            scale = 2.0 / mesh.extents.max()
            mesh.apply_scale(scale)

        points, sdf = sample_sdf_near_surface(mesh, number_of_points=n)
        sdf = sdf.astype(np.float32)[:, np.newaxis]

        return {
            'points': points.astype(np.float32),
            'distances': sdf.astype(np.float32)
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