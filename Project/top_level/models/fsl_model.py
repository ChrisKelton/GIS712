import torch.nn as nn
import torch
from typing import Optional, Callable, Union
import itertools
import numpy as np
from third_party.SEG_GRAD_CAM.gradcam_unet import GradCam
from Project.top_level.utils.torch_model import freeze_layers


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

        if train:
            self.backbone.train()
            n_params_in_encoder = 108
            n_params_to_not_freeze = len(list(self.backbone.parameters())) - n_params_in_encoder
            self.backbone = freeze_layers(self.backbone, n_layers_to_not_freeze=n_params_to_not_freeze)

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

    def get_label_from_target_labels_map(self, key: int) -> int:
        for key_set, val in self.target_labels_map.items():
            if key in key_set:
                return val

        raise IndexError(f"'{key}' not in 'self.target_labels_map'")

    def pseudo_label_generator(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > 1:
            raise NotImplementedError(f"Batch > 1. Not Implemented.")
        # with GradCAM(
        #     self.backbone,
        #     target_layer=self.backbone.child_image_classifier.model.layers[-1],
        #     input_shape=x.shape[1:],
        # ) as cam_extractor:
        #     out = self.backbone(x, normalize=True)
        #     activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        #
        # return activation_map

        # grad_cam = GradCam(
        #     model=self.backbone,
        #     feature_module=self.backbone.child_image_classifier.model.layers[-2].layers[1][0],
        #     # target_layer_names=[self.backbone.child_image_classifier.model.layers[-2].name],
        #     target_layer_names=[self.backbone.child_image_classifier.model.layers[-2].layers[1][0]],
        #     use_cuda=True if self.device == torch.device("cuda") else False,
        # )
        grad_cam = GradCam(
            model=self.backbone.child_image_classifier.model,
            # feature_module=self.backbone.child_image_classifier.model.layers[-2],
            # feature_module=self.backbone.child_image_classifier.model.layers[-1],
            feature_module=nn.ModuleList([self.feature_layer.to(self.device)]),
            # target_layer_names=[self.backbone.child_image_classifier.model.layers[-2].name],
            # target_layer_names=[self.backbone.child_image_classifier.model.layers[-1][0]],
            target_layer_names=["0"],
            use_cuda=True if self.device == torch.device("cuda") else False,
        )
        target_index = None
        mask = grad_cam(self.backbone.preprocess_image(x), target_index, activations_per_class=True)
        mask = torch.where(torch.isnan(torch.Tensor(mask)), 0.0, torch.Tensor(mask)).unsqueeze(0)
        mask = torch.argmax(mask, dim=1).unsqueeze(0).to(torch.float32).to(self.device)

        # mask_out = torch.zeros(x.shape[0], len(self.target_labels_map), x.shape[2:])
        # for key_set, target_label in self.target_labels_map.items():
        #     for key in key_set:
        #         mask_out[target_label] += torch.Tensor(mask[key])
        #     mask_out[target_label] /= len(key_set)
        # mask_out = torch.where(mask_out >= 0.5, 1.0, 0.0)

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
                # with torch.no_grad():
                # don't want to accumulate gradient during this, so figure out how to either turn it off or zero out the gradients
                # couldn't run with `torch.no_grad()` b/c the CAM activation function registers forward hooks that need the gradient for some reason
                if pseudo_label_generator is None:
                    pseudo_label_generator = self.pseudo_label_generator
                label = pseudo_label_generator(img)

            if len(label.shape) == 3:
                label = label.unsqueeze(1)
            # mask_features = torch.zeros((pixel_wise_features.shape[0], self.n_target_classes, *pixel_wise_features.shape[1:])).to(img.device)
            embedded_masked_features = torch.zeros((pixel_wise_features.shape[0], self.n_target_classes, *self.embedded_features_shape)).to(img.device)
            for classes_in_label, target_label in self.target_labels_map.items():
                label_by_class = torch.zeros_like(label)
                for class_ in classes_in_label:
                    label_by_class += torch.where(label == class_, label, -1.0)
                labels_by_class = torch.cat([label_.expand(13, *label_.shape[1:]).unsqueeze(0) for label_ in label_by_class])
                with torch.no_grad():
                    masked_features = self.backbone.pixel_wise_features(labels_by_class, normalize=False)
                    masked_features = torch.where(masked_features >= self.mask_features_th, 1.0, 0.0)
                    # mask_features[:, target_label] = self.backbone.pixel_wise_features(labels_by_class, normalize=False)
                    # mask_features[:, target_label] = torch.where(mask_features[:, target_label] >= self.mask_features_th, 1.0, 0.0)

                masked_features = torch.multiply(pixel_wise_features.clone(), masked_features)

                embedded_masked_features[:, target_label] = self.backbone.run_encoder_head(masked_features)
                # mask_features[:, target_label] = torch.multiply(pixel_wise_features.clone(), mask_features[:, target_label])
                #
                # mask_features[:, target_label] = self.backbone.run_encoder_head(mask_features[:, target_label])

            return embedded_masked_features
        else:
            return self.backbone(img, normalize=True, only_encoding=False)
        #     embedded_pixel_wise_features = self.encoder(pixel_wise_features)
        #
        # prediction = self.classifier_head(embedded_pixel_wise_features)
        # return prediction, embedded_pixel_wise_features