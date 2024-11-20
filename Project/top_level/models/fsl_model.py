import torch.nn as nn
import torch
from typing import Optional, Callable, Union
import itertools
import numpy as np
from third_party.SEG_GRAD_CAM.gradcam_unet import GradCam
from Project.top_level.utils.torch_model import freeze_layers, freeze_any_layers, unfreeze_any_layers
from contextlib import contextmanager


Device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FslSemSeg(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        n_target_classes: int,
        target_labels: list[int],
        test_img: torch.Tensor,
        target_labels_map: Optional[dict[int, int]] = None,
        mask_features_th: float = 0.1,
        device: Optional[torch.device] = None,
        train: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        out_channels = self.backbone.child_image_classifier.model.layers[-1][0].out_channels
        self.feature_layer = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

        n_params_in_encoder = 108  # len(self.backbone.encoder.parameters())
        if train:
            self.backbone.train()
            n_params_to_not_freeze = len(list(self.backbone.parameters())) - n_params_in_encoder
            self.backbone = freeze_layers(self.backbone, n_layers_to_not_freeze=n_params_to_not_freeze)
        self.n_params_in_encoder = n_params_in_encoder
        n_params_in_encoder_head = 2  # len(self.backbone.encoder_head.parameters())
        self.encoder_head_params_idx = list(
            np.arange(self.n_params_in_encoder, self.n_params_in_encoder + n_params_in_encoder_head)
        )

        if device is None:
            device = Device
        self.device = device

        with torch.no_grad():
            if len(test_img.shape) == 3:
                test_img = test_img.unsqueeze(0)
            test_embedded_features = self.backbone.run_encoder_head(self.backbone.pixel_wise_features(test_img))

        self.embedded_features_shape = test_embedded_features.shape[-3:]

        self.n_target_classes = n_target_classes
        self.target_labels = target_labels
        if target_labels_map is None:
            target_labels_map = {}
            for label in target_labels:
                target_labels_map[(label,)] = label
        else:
            grouped_keys: list[tuple[int, int]] = []
            grouped_vals: list[int] = []
            for key0, key1 in itertools.combinations(target_labels_map.keys(), r=2):
                label0 = target_labels_map[key0]
                label1 = target_labels_map[key1]
                if label0 == label1:
                    grouped_keys.append((key0, key1))
                    grouped_vals.append(label0)

            if len(grouped_keys) > 0:
                vals, cnts = np.unique(grouped_vals, return_counts=True)
                multiple_cnts_idx = np.where(cnts > 1)[0]

                grouped_keys_set: list[tuple[int, ...]] = []
                grouped_vals_set: list[int] = []
                for idx in multiple_cnts_idx:
                    grouped_key_set: set[int] = set()
                    val = vals[idx]
                    grouped_vals_idx = np.where(grouped_vals == val)[0]
                    for grouped_val_idx in grouped_vals_idx:
                        grouped_key_set = grouped_key_set.union(set(grouped_keys[grouped_val_idx]))
                    grouped_keys_set.append(tuple(grouped_key_set))
                    grouped_vals_set.append(val)

                for keys_set in grouped_keys_set:
                    for key in keys_set:
                        target_labels_map.pop(key)

                new_target_label_map = {}
                for key, val in target_labels_map.items():
                    new_target_label_map[(key,)] = val
                target_labels_map = new_target_label_map

                for key_set, val in zip(grouped_keys_set, grouped_vals_set):
                    target_labels_map[key_set] = val
            else:
                tmp: dict[tuple[int, ...], int] = {}
                for key, val in target_labels_map.items():
                    tmp[(key,)] = val
                target_labels_map = tmp.copy()

        self.target_labels_map: dict[tuple[int, ...], int] = target_labels_map.copy()
        self.mask_features_th = mask_features_th

    def _freeze_encoder_head(self):
        freeze_any_layers(self.backbone, self.encoder_head_params_idx)

    def _unfreeze_encoder_head(self):
        unfreeze_any_layers(self.backbone, self.encoder_head_params_idx)

    @contextmanager
    def freeze_encoder_head(self):
        self._freeze_encoder_head()
        yield
        self._unfreeze_encoder_head()

    def get_label_from_target_labels_map(self, key: int) -> int:
        for key_set, val in self.target_labels_map.items():
            if key in key_set:
                return val

        raise IndexError(f"'{key}' not in 'self.target_labels_map'")

    def pseudo_label_generator(self, x: torch.Tensor) -> torch.Tensor:
        grad_cam = GradCam(
            model=self.backbone.child_image_classifier.model,
            feature_module=nn.ModuleList([self.feature_layer.to(self.device)]),
            target_layer_names=["0"],
            use_cuda=True if self.device == torch.device("cuda") else False,
        )
        target_index = None
        # TODO: revert changes in SEG_GRAD_CAM fork and separate CAM output correctly
        mask = grad_cam(self.backbone.preprocess_image(x), target_index, activations_per_class=True)
        mask = torch.where(torch.isnan(torch.Tensor(mask)), 0.0, torch.Tensor(mask))
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        mask = torch.argmax(mask, dim=1)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(torch.float32).to(self.device)

        return mask

    def forward(
        self,
        img: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        pseudo_label_generator: Optional[Callable] = None,
        mask_pixel_wise_features: bool = True,
        do_prediction: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if mask_pixel_wise_features:
            pixel_wise_features = self.backbone.pixel_wise_features(img, normalize=True)

            if label is None:
                if pseudo_label_generator is None:
                    pseudo_label_generator = self.pseudo_label_generator
                label = pseudo_label_generator(img)

            if len(label.shape) == 3:
                label = label.unsqueeze(1)
            embedded_masked_features = torch.zeros((pixel_wise_features.shape[0], self.n_target_classes, *self.embedded_features_shape)).to(img.device)
            for classes_in_label, target_label in self.target_labels_map.items():
                label_by_class = torch.zeros_like(label)
                for class_ in classes_in_label:
                    label_by_class += torch.where(label == class_, label, -1.0)
                labels_by_class = torch.cat([label_.expand(13, *label_.shape[1:]).unsqueeze(0) for label_ in label_by_class])
                with torch.no_grad():
                    masked_features = self.backbone.pixel_wise_features(labels_by_class, normalize=False)
                    masked_features = torch.where(masked_features >= self.mask_features_th, 1.0, 0.0)

                masked_features = torch.multiply(pixel_wise_features.clone(), masked_features)

                embedded_masked_features[:, target_label] = self.backbone.run_encoder_head(masked_features)

            return embedded_masked_features
        else:
            return self.backbone(img, normalize=True, only_encoding=False)
