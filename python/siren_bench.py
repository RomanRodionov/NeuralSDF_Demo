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

def train(config_path, hidden_layers=4, hidden_features=32):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Building dataset...")

    dataset = SDF_Dataset(
        dataset_size=config["dataset_size"],
        batch_size=config["batch_size"],
        model_path=config["model_path"],
        checkpoint=config.get("checkpoint", "checkpoints/sdf_data.npy"),
        normalize=config.get("normalize", True),
        uniform_ratio=config["uniform_ratio"]
    )
    dataset.cuda()

    model = Siren(
            in_features=3,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=1,
            outermost_linear=True,
            first_omega_0=config.get("first_omega_0", 30),
            hidden_omega_0=config.get("hidden_omega_0", 30)
        ).to(device)

    params_count = total_parameters(model.net)
    print(f"Model has {params_count} parameters")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    lambda1 = lambda epoch: config["lambda_lr"] ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    progressbar = tqdm.tqdm(range(config["epochs"]))
    eikonal_weight = config["eikonal_weight"]
    heat_weight = config["heat_weight"]

    for epoch in progressbar:
        dataset.shuffle()
        total_loss = 0.0
        for i in range(dataset.n_batches()):
            batch = dataset.get_batch(i)
            points = batch["points"].to(device)
            sdf_gt = batch["dist"].to(device)
            #print(torch.max(points), torch.min(points))

            optimizer.zero_grad()
            sdf_pred, coords = model(points)

            if eikonal_weight > 0.0 or heat_weight > 0.0:
                grads = torch.autograd.grad(
                    outputs=sdf_pred,
                    inputs=coords,
                    grad_outputs=torch.ones_like(sdf_pred),
                    create_graph=True,
                    retain_graph=True,
                )[0]

            eikonal_val = eikonal_weight * eikonal_loss(sdf_pred, coords, grads) if eikonal_weight > 0.0 else 0
            heat_val = heat_weight * heat_loss(coords, sdf_pred, grads) if heat_weight > 0.0 else 0

            loss = criterion(sdf_pred, sdf_gt) + eikonal_val + heat_val
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        scheduler.step()

        progressbar.write(f"[Epoch {epoch+1}/{config['epochs']}] Loss: {total_loss:.6f}")

    save_path = f"checkpoints/siren_weights_{hidden_layers}_{hidden_features}.pth"
    save_model(model, save_path)
    save_raw = f"checkpoints/raw_weights_{hidden_layers}_{hidden_features}.bin"
    model.save_raw(save_raw)
    print(f"Model saved to {save_path}")

    print("Extracting mesh...")
    mesh = reconstruct_sdf(model, resolution=256)
    extracted_path = f"reconstructed_{hidden_layers}_{hidden_features}.obj"
    mesh.export(extracted_path)
    print(f"Mesh extracted to {extracted_path}")

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    CONFIG_PATH = "python/config.json"

    hidden_layers = [1, 2, 3, 4]
    hidden_features = [16, 32, 64, 128]

    for l in hidden_layers:
        for f in hidden_features:
            print(f"Train Siren: layers={l}, width={f}")
            train(CONFIG_PATH, l, f)