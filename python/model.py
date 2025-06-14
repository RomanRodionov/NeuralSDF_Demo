import torch
import torch.nn as nn
import numpy as np

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