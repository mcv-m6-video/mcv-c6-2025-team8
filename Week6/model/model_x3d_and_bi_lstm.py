"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F


#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d) #B, T, D

            #MLP
            im_feat = self._fc(im_feat) #B, T, num_classes+1

            return im_feat 
        
        def normalize(self, x):
            return x / 255
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))

    class X3D(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self.num_classes = args.num_classes

            # Load pretrained X3D backbone
            self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)

            # Extract feature dimension and remove the final classifier
            self.feat_dim = self.backbone.blocks[-1].proj.in_features
            self.backbone.blocks[-1].proj = nn.Identity()

            # Temporal modeling via bidirectional LSTM
            self.temporal_model = nn.LSTM(
                input_size=self.feat_dim,
                hidden_size=256,
                batch_first=True,
                bidirectional=True
            )

            # Frame-wise classifier head
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes + 1)  # +1 for background class
            )

            # Data augmentations
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.8, contrast=0.8)], p=0.3),
                T.RandomHorizontalFlip(),
            ])

            # ImageNet standard normalization
            self.standarization = T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )

            # Freeze all backbone parameters except the final block
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.blocks[-1].parameters():
                param.requires_grad = True

        def forward(self, x):
            # Input shape: [B, T, C, H, W]
            x = self.normalize(x)

            if self.training:
                x = self.augment(x)

            x = self.standarize(x)
            x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]

            # Extract features: [B, D, 1, 1, 1] -> [B, D]
            feat = self.backbone(x).view(x.size(0), self.feat_dim)

            # Repeat features across temporal dimension: [B, T, D]
            T_len = x.size(2)
            feat = feat.unsqueeze(1).repeat(1, T_len, 1)

            # Temporal modeling: [B, T, D] -> [B, T, 512]
            feat, _ = self.temporal_model(feat)

            # Classification: [B, T, 512] -> [B, T, num_classes + 1]
            out = self.classifier(feat)
            return out

        def normalize(self, x):
            return x / 255.

        def augment(self, x):
            for i in range(x.size(0)):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.size(0)):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            total = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'Model params: {total}')
            
        
    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.X3D(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()
