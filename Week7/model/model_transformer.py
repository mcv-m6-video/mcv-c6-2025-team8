"""
File containing the main model using Transformer.
"""

# Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim
            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features

            # Transformer for temporal modeling
            self.transformer = nn.TransformerEncoderLayer(
                d_model=self._d,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )

            # Final classification
            self._fc = FCLayers(self._d, args.num_classes + 1)

            # Augmentations
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            x = self.normalize(x)
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x)

            x = self.standarize(x)

            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)  # (B, T, D)

            # Transformer
            im_feat = self.transformer(im_feat)  # (B, T, D)

            # Final classification
            im_feat = self._fc(im_feat)  # (B, T, num_classes + 1)

            return im_feat

        def normalize(self, x):
            return x / 255.

        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:', sum(p.numel() for p in self.parameters()))

    class X3D(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.model.blocks[-1].proj = nn.Linear(2048, args.num_classes)

            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])
            self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            def forward(self, x):
                x = self.normalize(x)
                if self.training:
                    x = self.augment(x)
                x = self.standarize(x)
                x = x.permute(0, 2, 1, 3, 4)
                return self.model(x)

            def normalize(self, x):
                return x / 255.

            def augment(self, x):
                for i in range(x.shape[0]):
                    x[i] = self.augmentation(x[i])
                return x

            def standarize(self, x):
                for i in range(x.shape[0]):
                    x[i] = self.standarization(x[i])
                return x

    def __init__(self, args=None):
        self.device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        inference = optimizer is None
        self._model.eval() if inference else self._model.train()

        weights = torch.tensor([1.0] + [5.0] * self._num_classes, dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if inference else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1)
                    label = label.view(-1)
                    loss = F.cross_entropy(pred, label, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        seq = seq.to(self.device).float()

        self._model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred = self._model(seq)
            pred = torch.softmax(pred, dim=-1)
            return pred.cpu().numpy()
