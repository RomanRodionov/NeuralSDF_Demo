import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
    def __init__(self, 
                 in_features=3, hidden_features=256, 
                 hidden_layers=3, out_features=1, outermost_linear=True, 
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
        output = self.net(coords - 1)
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
        
    def load_raw(self, path, device=None):
        """
        Load weights from a raw file and move them to the specified device.
        
        Args:
            path (str): Path to the raw weights file.
            device (str or torch.device, optional): Target device for the model.
                If None, uses the device of the first parameter (or CPU if none).
        """
        with open(path, "rb") as f:
            # Read number of layers
            num_layers_bytes = f.read(4)
            if len(num_layers_bytes) < 4:
                raise ValueError("File too short: missing layer count")
            num_layers = int.from_bytes(num_layers_bytes, "little")

            # Extract current model's linear layers in the same order as save_raw
            current_layers = []
            for layer in self.net.children():
                if isinstance(layer, nn.Linear):
                    current_layers.append(layer)
                elif isinstance(layer, SineLayer):
                    current_layers.append(layer.linear)

            if len(current_layers) != num_layers:
                raise ValueError(f"Model has {len(current_layers)} linear layers, but file has {num_layers}")

            # Determine target device
            if device is None:
                # Use the device of the first parameter if available, else CPU
                params = list(self.parameters())
                device = params[0].device if params else torch.device('cpu')
            else:
                device = torch.device(device)

            # Move entire model to target device before loading (optional but safe)
            self.to(device)

            for i, layer in enumerate(current_layers):
                # Read dimensions
                out_dim_bytes = f.read(4)
                in_dim_bytes = f.read(4)
                if len(out_dim_bytes) < 4 or len(in_dim_bytes) < 4:
                    raise ValueError(f"File too short: missing dimensions for layer {i}")
                out_dim = int.from_bytes(out_dim_bytes, "little")
                in_dim = int.from_bytes(in_dim_bytes, "little")

                # Read weight data
                weight_size = out_dim * in_dim * 4  # float32 = 4 bytes
                weight_bytes = f.read(weight_size)
                if len(weight_bytes) < weight_size:
                    raise ValueError(f"File too short: missing weight data for layer {i}")
                weight_np = np.frombuffer(weight_bytes, dtype=np.float32).reshape(out_dim, in_dim)
                weight_tensor = torch.from_numpy(weight_np)

                # Read bias data
                bias_size = out_dim * 4
                bias_bytes = f.read(bias_size)
                if len(bias_bytes) < bias_size:
                    raise ValueError(f"File too short: missing bias data for layer {i}")
                bias_np = np.frombuffer(bias_bytes, dtype=np.float32).reshape(out_dim)
                bias_tensor = torch.from_numpy(bias_np)

                # Check shape compatibility
                if layer.weight.shape != weight_tensor.shape:
                    raise ValueError(f"Layer {i} weight shape mismatch: expected {layer.weight.shape}, got {weight_tensor.shape}")
                if layer.bias.shape != bias_tensor.shape:
                    raise ValueError(f"Layer {i} bias shape mismatch: expected {layer.bias.shape}, got {bias_tensor.shape}")

                # Assign weights (tensors are moved to the layer's device automatically via .to())
                layer.weight.data = weight_tensor.to(layer.weight.device)
                layer.bias.data = bias_tensor.to(layer.bias.device)

        print(f"Successfully loaded weights from {path} to device {device}")

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
    else:
        return None