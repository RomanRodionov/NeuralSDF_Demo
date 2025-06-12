import torch
import trimesh
from skimage import measure
import os

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def reconstruct_sdf(model, resolution=128, bound=1.0, device='cuda'):
    # reconstructs mesh in [-bound, bound]

    lin = torch.linspace(-bound, bound, resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)

    sdf_values = []
    batch_size = 65536
    with torch.no_grad():
        for i in range(0, coords.shape[0], batch_size):
            pred, _ = model(coords[i:i+batch_size])
            sdf_values.append(pred.squeeze(-1).cpu())
    
    sdf_values = torch.cat(sdf_values, dim=0).numpy()
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)

    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0)

    verts -= resolution / 2
    verts *= (2 * bound / resolution)

    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
