import torch
from torch import nn
import timm
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm


# Local imports
from model.modules import BaseRGBModel, FCLayers, step
import torchvision.transforms as T

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

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features

            # Modified MLP for classification and offsets (start, end)
            # +2 for the start and end temporal offsets
            self._fc = FCLayers(self._d, args.num_classes + 3)  # +3: 1 for background, 2 for offsets

            # Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            # Standardization
            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x)  # Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape  # B, T, C, H, W

            if self.training:
                x = self.augment(x)  # Augmentation per-batch

            x = self.standarize(x)  # Standardization (ImageNet stats)

            im_feat = self._features(x.view(-1, channels, height, width)).reshape(batch_size, clip_len, self._d)  # B, T, D

            # MLP for classification and temporal offset predictions
            im_feat = self._fc(im_feat)  # B, T, num_classes+3 (class + start offset + end offset)
            class_preds = im_feat[:, :, :-2]  # All but the last two columns for class predictions
            pred_offsets = im_feat[:, :, -2:]
            # print("class_pred and pred_offsets: ",class_preds.shape, pred_offsets.shape)
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
            print('Model params:', sum(p.numel() for p in self.parameters()))


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
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                # Here we also assume that `batch['start_offset']` and `batch['end_offset']`
                # contain the ground truth offsets for each action.
                start_offset = batch['start_offset'].to(self.device).float()
                end_offset = batch['end_offset'].to(self.device).float()
                
                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    #print(pred.shape)
                    pred_class = pred[..., :self._num_classes + 1]  # Class predictions
                    pred_offsets = pred[..., -2:]  # Temporal offsets (start, end)
                    
                    # print(pred_class.shape)
                    # print(pred_offsets.shape)
                    
                    label = label.view(-1)  # B*T
                    # pred_offsets_flat = pred_offsets.view(-1, 2)
                    # print([start_offset.shape, end_offset.shape])
                    
                    classification_loss = F.cross_entropy(pred_class.view(-1, self._num_classes + 1), label)
                    # print(pred_offsets.shape)
                    # print(100 * '_')
                  

                    pred_offsets = pred_offsets.view(-1, 2)  # Ensure pred_offsets is in shape (batch_size, 2)
                    target_offsets = torch.stack([start_offset, end_offset], dim=-1).view(-1, 2)  # Ensure target is in shape (batch_size, 2)
                    # print(pred_offsets.shape)
                    # print(target_offsets.shape)
                    # print(pred_offsets.shape)

                    # Now calculate the loss
                    offset_loss = F.smooth_l1_loss(pred_offsets, target_offsets)

                    # Temporal offset loss (MSE or SmoothL1)
                    offset_loss = F.smooth_l1_loss(pred_offsets.view(-1, 2), torch.stack([start_offset, end_offset], dim=-1).view(-1, 2))

                    loss = classification_loss + 1.0 * offset_loss  # Combining both

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)  # Avg loss

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
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # Apply softmax for class predictions
            class_preds = torch.softmax(pred[..., :self._num_classes + 1], dim=-1)
            offsets_preds = pred[..., -2:]

            return class_preds.cpu().numpy(), offsets_preds.cpu().numpy()
