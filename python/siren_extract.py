import torch
import torch.nn as nn
import torch.optim as optim
import json

from dataset import *
from model_siren import Siren
from utils import *
from torch_utils import *

import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = Siren().cuda()
    model.load_raw("model_current.bin")

    print("Extracting mesh...")
    mesh = reconstruct_sdf(model, resolution=512)
    extracted_path = f"reconstructed.obj"
    mesh.export(extracted_path)
    print(f"Mesh extracted to {extracted_path}")