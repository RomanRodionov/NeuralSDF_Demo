import torch
import torch.nn as nn
import torch.optim as optim
import json

from dataset import *
from model import *
from utils import *

import tqdm

def train(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Building dataset...")

    dataset = SDF_Dataset(
        dataset_size=config["dataset_size"],
        batch_size=config["batch_size"],
        model_path=config["model_path"],
        checkpoint=config.get("checkpoint", "checkpoints/sdf_data.npy"),
        normalize=config.get("normalize", True)
    )
    dataset.cuda()

    model = Siren(
        in_features=3,
        hidden_features=config["hidden_features"],
        hidden_layers=config["hidden_layers"],
        out_features=1,
        outermost_linear=True,
        first_omega_0=config.get("first_omega_0", 30),
        hidden_omega_0=config.get("hidden_omega_0", 30)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    progressbar = tqdm.tqdm(range(config["epochs"]))

    for epoch in progressbar:
        dataset.shuffle()
        total_loss = 0.0
        for i in range(dataset.n_batches()):
            batch = dataset.get_batch(i)
            points = batch["points"].to(device)
            sdf_gt = batch["dist"].to(device)

            optimizer.zero_grad()
            sdf_pred, _ = model(points)

            loss = criterion(sdf_pred, sdf_gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        progressbar.write(f"[Epoch {epoch+1}/{config['epochs']}] Loss: {total_loss:.6f}")

    save_path = config.get("save_path", "checkpoints/siren_weights.pth")
    save_model(model, save_path)
    print(f"Model saved to {save_path}")

def extract_mesh(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Siren(
        in_features=3,
        hidden_features=config["hidden_features"],
        hidden_layers=config["hidden_layers"],
        out_features=1,
        outermost_linear=True,
        first_omega_0=config.get("first_omega_0", 30),
        hidden_omega_0=config.get("hidden_omega_0", 30)
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/siren_weights.pth"))
    model.cuda().eval()

    mesh = reconstruct_sdf(model, resolution=128)
    mesh.export("reconstructed.obj")

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

    CONFIG_PATH = "python/config.json"

    train(CONFIG_PATH)
    extract_mesh(CONFIG_PATH)