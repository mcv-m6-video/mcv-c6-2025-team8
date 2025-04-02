import os
import json
import torch
import pickle
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import wandb
import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from dataset.frame import FrameReader
import torch.nn.functional as F
from tabulate import tabulate

# -------------------- Dataset -------------------- #

class TrackingEnhancedDataset(Dataset):
    def __init__(self, frame_dir, tracking_pt, label_pkl, frame_paths_pkl,
                 clip_len=50, stride=2, class_dict=None):
        self.frame_dir = frame_dir
        self.clip_len = clip_len
        self.stride = stride
        self.class_dict = class_dict

        with open(frame_paths_pkl, 'rb') as f:
            self.frame_paths = pickle.load(f)
        with open(label_pkl, 'rb') as f:
            self.label_store = pickle.load(f)

        # Preloaded tracking features
        self.tracking_tensor = torch.load(tracking_pt)

        # Frame reader wie im Baseline
        self.frame_reader = FrameReader(frame_dir, dataset="soccernetball")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_info = self.frame_paths[idx]
        labels_info = self.label_store[idx]

        # Load frame tensor using same method as baseline
        frames = self.frame_reader.load_frames(frame_info, pad=True, stride=self.stride).float()

        # Tracking features
        tracking_features = self.tracking_tensor[idx].float()

        # Multi-label binarized vector
        labels = torch.zeros(len(self.class_dict))
        for label in labels_info:
            labels[label['label'] - 1] = 1

        return {
            "frame": frames,  # shape (T, C, H, W)
            "tracking": tracking_features,  # shape (T, 4)
            "label": labels
        }


# -------------------- Model -------------------- #

class TrackingEnhancedModel(nn.Module):
    def __init__(self, num_classes, feature_arch="rny002"):
        super().__init__()
        import timm
        self.features = timm.create_model("regnety_002", pretrained=True)
        self.feat_dim = self.features.head.fc.in_features
        self.features.head.fc = nn.Identity()

        self.tracking_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

        self.norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x, tracking):
        B, T, C, H, W = x.shape
        x = x / 255.0
        for i in range(B):
            for j in range(T):
                x[i, j] = self.norm(x[i, j])

        x = x.view(-1, C, H, W)
        feat = self.features(x).view(B, T, -1)

        track_feat = self.tracking_proj(tracking)
        combined = torch.cat([feat, track_feat], dim=-1)
        pooled = torch.max(combined, dim=1)[0]
        return self.classifier(pooled)

# -------------------- Training -------------------- #

def train_epoch(model, loader, optimizer, device, scaler=None, scheduler=None):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        x = batch['frame'].to(device)
        t = batch['tracking'].to(device)
        y = batch['label'].to(device).float()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(x, t)
            loss = F.binary_cross_entropy_with_logits(pred, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    scores, targets = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            x = batch['frame'].to(device)
            t = batch['tracking'].to(device)
            y = batch['label'].to(device).float()

            logits = model(x, t)
            probs = torch.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            scores.append(probs.cpu().numpy())
            targets.append(y.cpu().numpy())
            total_loss += loss.item()

    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    ap = average_precision_score(targets, scores, average=None)
    return ap, total_loss / len(loader)

# -------------------- Main -------------------- #

def run_training(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    wandb.init(project="C6_W5", entity="C3_MCV_LGVP", config=cfg, name="baseline_with_yolo_tracking")

    device = torch.device(cfg['device'])
    class_dict = {str(i): i for i in range(1, cfg['num_classes'] + 1)}
    base_split = os.path.join(cfg['save_dir'], 'splits')

    def make_dataset(split):
        path_base = os.path.join(base_split, f"LEN{cfg['clip_len']}SPLIT{split}")
        return TrackingEnhancedDataset(
            frame_dir=cfg['frame_dir'],
            tracking_pt=os.path.join(path_base, 'tracking.pt'),
            label_pkl=os.path.join(path_base, 'labels.pkl'),
            frame_paths_pkl=os.path.join(path_base, 'frame_paths.pkl'),
            clip_len=cfg['clip_len'],
            stride=cfg['stride'],
            class_dict=class_dict
        )

    train_ds = make_dataset("train")
    val_ds = make_dataset("val")
    test_ds = make_dataset("test")

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    model = TrackingEnhancedModel(cfg['num_classes'], cfg['feature_arch']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_loss = float('inf')
    for epoch in range(cfg['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['num_epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        ap, val_loss = evaluate(model, val_loader, device)
        avg_ap = np.mean(ap) * 100

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {avg_ap:.2f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mAP": avg_ap})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], "checkpoint_best.pt"))

    # Final evaluation
    model.load_state_dict(torch.load(
        os.path.join(cfg['save_dir'], "yolo", "checkpoints", "checkpoint_best.pt"),
        map_location=torch.device('cpu')
    ))

    for i, class_name in enumerate(class_dict.keys()):
        print("works")
    ap, _ = evaluate(model, test_loader, device)
    avg_map = np.mean(ap) * 100

    table = []
    for i, class_name in enumerate(class_dict.keys()):
        table.append([class_name, f"{ap[i]*100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    avg_map = np.mean(ap) * 100
    avg_table = [["Average", f"{avg_map:.2f}"]]
    headers = ["", "Average Precision"]
    print(tabulate(avg_table, headers, tablefmt="grid"))

    wandb.log({"Final mAP": avg_map})
    print(f"\nTest mAP: {avg_map:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training("config/baseline_with_yolo_tracking.json")
