import os
import json
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_dir', type=str, required=True)
    parser.add_argument('--tracking_dir', type=str, required=True)
    parser.add_argument('--index_file', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--clip_len', type=int, default=50)
    parser.add_argument('--stride', type=int, default=2)
    return parser.parse_args()


def load_tracking_dict(tracking_dir):
    print("Indexing tracking data...")
    tracking_lookup = {}
    for root, _, files in os.walk(tracking_dir):
        if 'Tracking.json' in files:
            json_path = os.path.join(root, 'Tracking.json')
            rel_path = os.path.relpath(root, tracking_dir)
            with open(json_path, 'r') as f:
                data = json.load(f)
            for entry in data:
                key = (rel_path, entry['frame'])
                tracking_lookup[key] = tracking_lookup.get(key, []) + [entry]
    return tracking_lookup


def process_clip(frame_dir, tracking_lookup, clip, clip_len, stride):
    base_path, start, pad_start, pad_end, _, length = clip
    rel_path = os.path.relpath(base_path, frame_dir)

    tracking_feats = []

    for i in range(length - pad_start - pad_end):
        frame_idx = start + i * stride
        key = (rel_path, frame_idx)

        tracking_feat = torch.zeros(4)
        if key in tracking_lookup:
            objects = [o for o in tracking_lookup[key] if o['class_id'] in [0, 32]]
            if objects:
                coords = np.array([o['bbox'] for o in objects])
                center_x = np.mean(coords[:, 0]) / 398
                center_y = np.mean(coords[:, 1]) / 224
                has_ball = 1.0 if any(o['class_id'] == 32 for o in objects) else 0.0
                num_players = len([o for o in objects if o['class_id'] == 0]) / 10.0
                tracking_feat = torch.tensor([center_x, center_y, has_ball, num_players])

        tracking_feats.append(tracking_feat)

    # pad if needed
    T_actual = len(tracking_feats)
    if T_actual < clip_len:
        tracking_feats.extend([torch.zeros(4) for _ in range(clip_len - T_actual)])
    else:
        tracking_feats = tracking_feats[:clip_len]

    return torch.stack(tracking_feats)  # T, 4


def main():
    args = parse_args()

    with open(args.index_file, 'rb') as f:
        frame_paths = pickle.load(f)

    tracking_lookup = load_tracking_dict(args.tracking_dir)

    print(f"Processing {len(frame_paths)} clips...")
    tracking_features = []

    for clip in tqdm(frame_paths):
        feats = process_clip(args.frame_dir, tracking_lookup, clip, args.clip_len, args.stride)
        tracking_features.append(feats)

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, 'tracking.pt')
    torch.save(tracking_features, save_path)
    print(f"Saved tracking features to {save_path}")


if __name__ == '__main__':
    main()
