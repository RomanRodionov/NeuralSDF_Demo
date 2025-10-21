import torch
import torch.nn as nn
import torch.optim as optim
import json

from dataset import *
from model import *
from utils import *
from torch_utils import *

import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config_path, model_type="SIREN"):#SIREN or FastKAN
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

    model = get_model(config, model_type).to(device)

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

    save_path = config.get("save_path", "checkpoints/siren_weights.pth")
    save_model(model, save_path)
    model.save_raw(config.get("raw_weights_path", "checkpoints/raw_weights.bin"))
    print(f"Model saved to {save_path}")

def extract_mesh(config_path, model_type="SIREN"):
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(config, model_type).to(device)

    model.load_state_dict(torch.load("checkpoints/siren_weights.pth"))
    model.cuda().eval()

    mesh = reconstruct_sdf(model, resolution=256)
    mesh.export(config["reconstruction_path"])

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    train(args.config)
    """
    MODEL_TYPE = "SIREN"#"FastKAN"#"NGLoD"# 
    CONFIG_PATH = "python/config.json"

    train(CONFIG_PATH, MODEL_TYPE)

    print("Extracting mesh...")
    extract_mesh(CONFIG_PATH, MODEL_TYPE)