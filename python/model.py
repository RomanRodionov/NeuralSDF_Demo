import torch
import torch.nn as nn
import numpy as np
from fastkan import FastKAN as KAN
import torch.nn.functional as F
import tinycudann

class PositionalEncoding(nn.Module):
    def __init__(self, m=6, sigma=2):
        super(PositionalEncoding, self).__init__()
        self.m = m
        self.sigma = sigma

    def __len__(self):
        return self.m * 2

    def forward(self, x):
        res = []
        for i in range(self.m):
            coef = 2 * torch.pi * self.sigma ** (i / self.m)
            res.append(torch.sin(coef * x))
            res.append(torch.cos(coef * x))

        return torch.cat(res, dim=-1)
    
class FrequencyEncoding(nn.Module):
    def __init__(self, m=12, sigma=2):
        super(FrequencyEncoding, self).__init__()
        self.m = m
        self.sigma = sigma

    def __len__(self):
        return self.m

    def forward(self, x):
        res = []
        for i in range(self.m):
            coef = 2 * torch.pi * self.sigma ** i
            res.append(torch.sin(coef * x))

        return torch.cat(res, dim=-1)
    
class HashGridEncoding(nn.Module):
    def __init__(
        self,
        range,
        dim=3,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=15,
        base_resolution=16,
        finest_resolution=512,
    ):
        super(HashGridEncoding, self).__init__()
        self.input_dim = dim
        b = (finest_resolution / base_resolution) ** (1 / (n_levels - 1))
        self.config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "finest_resolution": finest_resolution,
            "per_level_scale": b,
        }
        self.enc = tinycudann.Encoding(self.input_dim, self.config)
        self.range = range

    def __len__(self):
        return self.config["n_levels"] * self.config["n_features_per_level"]

    def forward(self, x):
        x = (x + self.range) / (2 * self.range)
        orig_shape = x.shape
        x = x.reshape(-1, self.input_dim)
        x = self.enc(x).float()
        x = x.reshape(*orig_shape[:-1], -1)
        return x

# Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020). 
# Implicit Neural Representations with Periodic Activation Functions. Proc. NeurIPS. 
# https://arxiv.org/abs/2006.09661

class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords
    
    def save_raw(self, path):
        with open(path, "wb") as f:
            #f.write("hydrann1".encode("utf-8"))
            layers = []
            for layer in self.net.children():
                if isinstance(layer, nn.Linear):
                    layers.append(layer)
                elif isinstance(layer, SineLayer):
                    layers.append(layer.linear)
                    
            f.write(len(layers).to_bytes(4, "little"))
            for layer in layers:
                weight = np.ascontiguousarray(layer.weight.cpu().detach().numpy(), dtype=np.float32)
                bias = np.ascontiguousarray(layer.bias.cpu().detach().numpy(), dtype=np.float32)

                f.write(weight.shape[0].to_bytes(4, "little"))
                f.write(weight.shape[1].to_bytes(4, "little"))
                f.write(weight.tobytes())
                f.write(bias.tobytes())

class FastKAN_SDF(nn.Module):
    def __init__(self, dim, nCps=4, grid_max=1., grid_min=-1., encoding=None):
        super().__init__()
        
        self.enc = encoding
        self.net = KAN(dim, num_grids=nCps, grid_max=grid_max, grid_min=grid_min, base_activation=F.relu, use_base_update=True)
    
    def forward(self, coords):
        if self.enc is not None:
            coords = torch.cat((self.enc(coords), coords), dim=-1)
        #coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)#, use_layernorm=False)
        return output, coords
    
    def save_raw(self, path):
        pass

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, encoding=None):
        super().__init__()
        
        self.enc = encoding
        self.net = []
        self.net.extend([nn.Linear(in_features, hidden_features), nn.ReLU()])
        for i in range(hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features), nn.ReLU()])
        self.net.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.enc is not None:
            coords = torch.cat((self.enc(coords), coords), dim=-1)
        #coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords
    
    def save_raw(self, path):
        with open(path, "wb") as f:
            #f.write("hydrann1".encode("utf-8"))
            layers = []
            for layer in self.net.children():
                if isinstance(layer, nn.Linear):
                    layers.append(layer)
                elif isinstance(layer, SineLayer):
                    layers.append(layer.linear)
                    
            f.write(len(layers).to_bytes(4, "little"))
            for layer in layers:
                weight = np.ascontiguousarray(layer.weight.cpu().detach().numpy(), dtype=np.float32)
                bias = np.ascontiguousarray(layer.bias.cpu().detach().numpy(), dtype=np.float32)

                f.write(weight.shape[0].to_bytes(4, "little"))
                f.write(weight.shape[1].to_bytes(4, "little"))
                f.write(weight.tobytes())
                f.write(bias.tobytes())


def get_model(config, model_type="SIREN"):
    if model_type == "SIREN":
        return Siren(
            in_features=3,
            hidden_features=config["hidden_features"],
            hidden_layers=config["hidden_layers"],
            out_features=1,
            outermost_linear=True,
            first_omega_0=config.get("first_omega_0", 30),
            hidden_omega_0=config.get("hidden_omega_0", 30)
        )
    elif model_type == "FastKAN":
        #encoding = PositionalEncoding(m=config["freq_enc_dim"])
        encoding = HashGridEncoding(1, log2_hashmap_size=config["log2_hashmap_size"])
        return FastKAN_SDF(
            dim=[len(encoding) + 3, *(config["kan_features"] for _ in range(config["kan_layers"])), 1],
            nCps=config["kan_grid"],
            encoding=encoding)
    elif model_type == "NGLoD":
        #encoding = PositionalEncoding(m=config["freq_enc_dim"])
        encoding = HashGridEncoding(1, log2_hashmap_size=config["log2_hashmap_size"])
        return MLP(
            in_features=len(encoding) + 3,
            hidden_features=config["hidden_features"],
            hidden_layers=config["hidden_layers"],
            out_features=1,
            encoding=encoding
        )