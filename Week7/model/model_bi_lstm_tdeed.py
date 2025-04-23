import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from model.modules import BaseRGBModel, FCLayers, step
import random  # >>> added for mixup

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch

            self._use_disp = getattr(args, 'use_displacement', False)  # >>> added: optional displacement head
            self._clip_len = args.clip_len  # >>> added: for positional encoding
            self._num_classes = args.num_classes + 1  # >>> added: unified num_classes
            self._mixup_alpha = getattr(args, 'mixup_alpha', 0.2)  # >>> added: mixup param

            # Feature extractor (RegNet)
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

            # >>> added: Temporal positional encoding
            self.temp_enc = nn.Parameter(
                torch.normal(mean=0, std=1 / args.clip_len, size=(args.clip_len, self._d))
            )

            # BiLSTM
            self.temporal_model = nn.LSTM(
                input_size=self._d,
                hidden_size=self._d,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.temporal_fc = nn.Linear(2 * self._d, self._d)

            # Classification head
            self._fc = FCLayers(self._d, self._num_classes)

            # >>> added: Displacement regression head (optional)
            if self._use_disp:
                self._pred_displ = FCLayers(self._d, 1)

            # Augmentation
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
            ])

        # >>> modified: forward now supports inference, amp, and mixup
        def forward(self, x, inference=False, use_amp=True, apply_mixup=False):
            x = self.normalize(x)
            B, T, C, H, W = x.shape

            if self.training:
                x = self.augment(x)

            x = self.standarize(x)

            # RegNet feature extraction
            x = x.view(-1, C, H, W)
            with torch.cuda.amp.autocast() if use_amp else nullcontext():  # >>> added: AMP support
                features = self._features(x)
            features = features.view(B, T, -1)

            # >>> added: temporal positional encoding
            features = features + self.temp_enc.unsqueeze(0)

            # >>> added: mixup in feature space
            if self.training and apply_mixup:
                l = [random.betavariate(self._mixup_alpha, self._mixup_alpha) for _ in range(B)]
                l = torch.tensor(l, device=features.device).view(-1, 1, 1)
                perm = torch.randperm(B)
                features = l * features + (1 - l) * features[perm]

            # BiLSTM
            features, _ = self.temporal_model(features)
            features = self.temporal_fc(features)

            # Classification head
            cls_logits = self._fc(features)

            # >>> added: optional displacement output
            if self._use_disp:
                disp_pred = self._pred_displ(features)
                return {'im_feat': cls_logits, 'displ_feat': disp_pred}

            return cls_logits

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
            print('Model params:',
                  sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
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
            for batch in tqdm(loader):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                with torch.cuda.amp.autocast(enabled=getattr(self._args, 'use_amp', True)):
                    output = self._model(frame, use_amp=getattr(self._args, 'use_amp', True),
                                         apply_mixup=getattr(self._args, 'mixup_alpha', 0) > 0)
                    if isinstance(output, dict):
                        pred = output['im_feat']
                    else:
                        pred = output

                    pred = pred.view(-1, self._num_classes + 1)
                    label = label.view(-1)
                    loss = F.cross_entropy(pred, label, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=getattr(self._args, 'use_amp', True)):
                output = self._model(seq, inference=True)
                if isinstance(output, dict):
                    output = output['im_feat']
                output = torch.softmax(output, dim=-1)
                return output.cpu().numpy()
