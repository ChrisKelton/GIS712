try:
    import os, json
    import numpy as np
    import torch
    import torch.nn as nn
    import math
    from torch import tensor
    from torchvision import transforms
    from arcgis.learn.models._inferencing import util
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

import arcgis
from arcgis.learn.models._unet import UnetClassifier

import importlib


def remap(tensor, idx2pixel):
    modified_tensor = torch.zeros_like(tensor)
    for id, pixel in idx2pixel.items():
        modified_tensor[tensor == id] = pixel
    return modified_tensor

def is_cont(class_values):
    flag = True
    for i in range(len(class_values) - 1):
        if class_values[i] + 1 != class_values[i+1]:
            flag = False
    return flag

def variable_tile_size_check(json_info, parameters):
    if json_info.get("SupportsVariableTileSize", False):
        parameters.extend(
            [
                {
                    "name": "tile_size",
                    "dataType": "numeric",
                    "value": int(json_info["ImageHeight"]),
                    "required": False,
                    "displayName": "Tile Size",
                    "description": "Tile size used for inferencing",
                }
            ]
        )
    return parameters

def dihedral_transform(x, k):  # expects [C, H, W]
    flips = []
    if k & 1:
        flips.append(1)
    if k & 2:
        flips.append(2)
    if flips:
        x = torch.flip(x, flips)
    if k & 4:
        x = x.transpose(1, 2)
    return x.contiguous()


def create_interpolation_mask(side, border, device, window_fn="bartlett"):
    if window_fn == "bartlett":
        window = torch.bartlett_window(side, device=device).unsqueeze(0)
        interpolation_mask = window * window.T
    elif window_fn == "hann":
        window = torch.hann_window(side, device=device).unsqueeze(0)
        interpolation_mask = window * window.T
    else:
        linear_mask = torch.linspace(0, 1, border, device=device).repeat(side, 1)
        remainder_tile = torch.ones((side, side - border), device=device)
        interp_tile = torch.cat([linear_mask, remainder_tile], dim=1)

        interpolation_mask = torch.ones((side, side), device=device)
        for i in range(4):
            interpolation_mask = interpolation_mask * interp_tile.rot90(i)
    return interpolation_mask


def unfold_tensor(tensor, tile_size, stride):  # expects tensor  [1, C, H, W]
    mask = torch.ones_like(tensor[0][0].unsqueeze(0).unsqueeze(0))

    unfold = nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    # Apply to mask and original image
    mask_p = unfold(mask)
    patches = unfold(tensor)

    patches = patches.reshape(tensor.size(1), tile_size, tile_size, -1).permute(
        3, 0, 1, 2
    )
    masks = mask_p.reshape(1, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    return masks, (tensor.size(2), tensor.size(3)), patches


def fold_tensor(input_tensor, masks, t_size, tile_size, stride):
    input_tensor_permuted = (
        input_tensor.permute(1, 2, 3, 0).reshape(-1, input_tensor.size(0)).unsqueeze(0)
    )
    mask_tt = masks.permute(1, 2, 3, 0).reshape(-1, masks.size(0)).unsqueeze(0)

    fold = nn.Fold(
        output_size=(t_size[0], t_size[1]),
        kernel_size=(tile_size, tile_size),
        stride=stride,
    )
    output_tensor = fold(input_tensor_permuted) / fold(mask_tt)
    return output_tensor


def calculate_rectangle_size_from_batch_size(batch_size):
    """
    calculate number of rows and cols to composite a rectangle given a batch size
    :param batch_size:
    :return: number of cols and number of rows
    """
    rectangle_height = int(math.sqrt(batch_size) + 0.5)
    rectangle_width = int(batch_size / rectangle_height)

    if rectangle_height * rectangle_width > batch_size:
        if rectangle_height >= rectangle_width:
            rectangle_height = rectangle_height - 1
        else:
            rectangle_width = rectangle_width - 1

    if (rectangle_height + 1) * rectangle_width <= batch_size:
        rectangle_height = rectangle_height + 1
    if (rectangle_width + 1) * rectangle_height <= batch_size:
        rectangle_width = rectangle_width + 1

    # swap col and row to make a horizontal rect
    if rectangle_height > rectangle_width:
        rectangle_height, rectangle_width = rectangle_width, rectangle_height

    if rectangle_height * rectangle_width != batch_size:
        return batch_size, 1

    return rectangle_height, rectangle_width


def get_tile_size(model_height, model_width, padding, batch_height, batch_width):
    """
    Calculate request tile size given model and batch dimensions
    :param model_height:
    :param model_width:
    :param padding:
    :param batch_width:
    :param batch_height:
    :return: tile height and tile width
    """
    tile_height = (model_height - 2 * padding) * batch_height
    tile_width = (model_width - 2 * padding) * batch_width

    return tile_height, tile_width


def tile_to_batch(
    pixel_block, model_height, model_width, padding, fixed_tile_size=True, **kwargs
):
    inner_width = model_width - 2 * padding
    inner_height = model_height - 2 * padding

    band_count, pb_height, pb_width = pixel_block.shape
    pixel_type = pixel_block.dtype

    if fixed_tile_size is True:
        batch_height = kwargs["batch_height"]
        batch_width = kwargs["batch_width"]
    else:
        batch_height = math.ceil((pb_height - 2 * padding) / inner_height)
        batch_width = math.ceil((pb_width - 2 * padding) / inner_width)

    batch = np.zeros(
        shape=(batch_width * batch_height, band_count, model_height, model_width),
        dtype=pixel_type,
    )
    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        # pixel block might not be the shape (band_count, model_height, model_width)
        sub_pixel_block = pixel_block[
            :,
            y * inner_height : y * inner_height + model_height,
            x * inner_width : x * inner_width + model_width,
        ]
        sub_pixel_block_shape = sub_pixel_block.shape
        batch[
            b, :, : sub_pixel_block_shape[1], : sub_pixel_block_shape[2]
        ] = sub_pixel_block

    return batch, batch_height, batch_width


def batch_to_tile(batch, batch_height, batch_width):
    batch_size, bands, inner_height, inner_width = batch.shape
    tile = np.zeros(
        shape=(bands, inner_height * batch_height, inner_width * batch_width),
        dtype=batch.dtype,
    )

    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        tile[
            :,
            y * inner_height : (y + 1) * inner_height,
            x * inner_width : (x + 1) * inner_width,
        ] = batch[b]

    return tile


class ChildImageClassifier(nn.Module):
    # def initialize(self, model, model_as_file):
    def __init__(self, model, model_as_file):
        super().__init__()

        if not HAS_TORCH:
            raise Exception(
                "PyTorch is not installed. Install it using conda install -c pytorch pytorch torchvision"
            )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if model_as_file:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        else:
            self.json_info = json.load(model)

        model_path = self.json_info["ModelFile"]
        if model_as_file and not os.path.isabs(model_path):
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(model), model_path)
            )

        # self.unet = UnetClassifier.from_emd(data=None, emd_path=model)
        unet = UnetClassifier.from_emd(data=None, emd_path=model)
        self.model = unet.learn.model.to(self.device)
        self.model.eval()

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                {
                    "name": "padding",
                    "dataType": "numeric",
                    "value": int(self.json_info["ImageHeight"]) // 4,
                    "required": False,
                    "displayName": "Padding",
                    "description": "Padding",
                },
                {
                    "name": "batch_size",
                    "dataType": "numeric",
                    "required": False,
                    "value": 4,
                    "displayName": "Batch Size",
                    "description": "Batch Size",
                },
                {
                    "name": "predict_background",
                    "dataType": "string",
                    "required": False,
                    "value": "True",
                    "displayName": "Predict Background",
                    "description": "If False, will never predict the background/NoData Class.",
                },
                {
                    "name": "test_time_augmentation",
                    "dataType": "string",
                    "required": False,
                    "value": "True",
                    "displayName": "Perform test time augmentation while predicting",
                    "description": "If True, will merge predictions from flipped and rotated images.",
                }
            ]
        )
        required_parameters = variable_tile_size_check(
            self.json_info, required_parameters
        )
        return required_parameters

    def getConfiguration(self, **scalars):
        self.tytx = int(scalars.get("tile_size", self.json_info["ImageHeight"]))
        self.padding = int(
            scalars.get("padding", self.tytx // 4)
        )  ## Default padding Imageheight//4.
        self.batch_size = (
            int(math.sqrt(int(scalars.get("batch_size", 4)))) ** 2
        )  ## Default 4 batch_size
        self.predict_background = scalars.get("predict_background", "true").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  ## Default value True

        (
            self.rectangle_height,
            self.rectangle_width,
        ) = calculate_rectangle_size_from_batch_size(self.batch_size)
        ty, tx = get_tile_size(
            self.tytx,
            self.tytx,
            self.padding,
            self.rectangle_height,
            self.rectangle_width,
        )

        self.use_tta = scalars.get("test_time_augmentation", "false").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  # Default value True

        return {
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "tx": tx,
            "ty": ty,
            "fixedTileSize": 1,
            "test_time_augmentation": self.use_tta,
        }

    def updatePixels(self, tlc, shape, props, **pixelBlocks):  # 8 x 224 x 224 x 3
        input_image = pixelBlocks["raster_pixels"].astype(np.float32)
        batch, batch_height, batch_width = tile_to_batch(
            input_image,
            self.tytx,
            self.tytx,
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )

        semantic_predictions = util.pixel_classify_image(
            self.model,
            batch,
            self.device,
            classes=[clas["Name"] for clas in self.json_info["Classes"]],
            predict_bg=self.predict_background,
            model_info=self.json_info,
        )
        semantic_predictions = batch_to_tile(
            semantic_predictions.unsqueeze(dim=1).cpu().numpy(),
            batch_height,
            batch_width,
        )
        return semantic_predictions

    def split_predict_interpolate(self, normalized_image_tensor):
        kernel_size = self.tytx
        stride = kernel_size - (2 * self.padding)

        # Split image into overlapping tiles
        masks, t_size, patches = unfold_tensor(
            normalized_image_tensor, kernel_size, stride
        )

        with torch.no_grad():
            output = self.model(patches)

        interpolation_mask = create_interpolation_mask(
            kernel_size, 0, self.device, "hann"
        )
        output = output * interpolation_mask
        masks = masks * interpolation_mask

        # merge predictions from overlapping chips
        int_surface = fold_tensor(output, masks, t_size, kernel_size, stride)

        return int_surface

    def tta_predict(self, normalized_image_tensor, test_time_aug=True):
        all_activations = []

        transforms = [0]
        if test_time_aug:
            if self.json_info["ImageSpaceUsed"] == "MAP_SPACE":
                transforms = list(range(8))
            else:
                transforms = [
                    0,
                    2,
                ]  # no vertical flips for pixel space (oriented imagery)

        for k in transforms:
            flipped_image_tensor = dihedral_transform(normalized_image_tensor[0], k)
            int_surface = self.split_predict_interpolate(
                flipped_image_tensor.unsqueeze(0)
            )
            corrected_activation = dihedral_transform(int_surface[0], k)

            if k in [5, 6]:
                corrected_activation = dihedral_transform(int_surface[0], k).rot90(
                    2, [1, 2]
                )

            all_activations.append(corrected_activation)

        all_activations = torch.stack(all_activations)

        return all_activations

    def updatePixelsTTA(self, tlc, shape, props, **pixelBlocks):  # 8 x 224 x 224 x 3
        model_info = self.json_info

        class_values = [clas["Value"] for clas in model_info["Classes"]]
        is_contiguous = is_cont([0] + class_values)

        if not is_contiguous:
            pixel_mapping = [0] + class_values
            idx2pixel = {i: d for i, d in enumerate(pixel_mapping)}

        input_image = pixelBlocks["raster_pixels"].astype(np.float32)
        input_image_tensor = torch.tensor(input_image).to(self.device).float()

        if "NormalizationStats" in model_info:
            normalized_image_tensor = util.normalize_batch(
                input_image_tensor.cpu(), model_info
            )
            normalized_image_tensor = normalized_image_tensor.float().to(
                input_image_tensor.device
            )
        else:
            normalize = transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            normalized_image_tensor = normalize(input_image_tensor / 255.0).unsqueeze(0)

        all_activations = self.tta_predict(
            normalized_image_tensor, test_time_aug=self.use_tta
        )
        # probability of each class in 2nd dimension
        all_activations = all_activations.mean(dim=0, keepdim=True)
        softmax_surface = all_activations.softmax(dim=1)

        ignore_mapped_class = model_info.get("ignore_mapped_class", [])
        predict_bg = True
        for k in ignore_mapped_class:
            softmax_surface[:, k] = -1

        if not predict_bg:
            softmax_surface[:, 0] = -1

        result = softmax_surface.max(dim=1)[1]

        if not is_contiguous:
            result = remap(result, idx2pixel)

        pad = self.padding

        return result.cpu().numpy().astype("i4")[:, pad : -pad or None, pad : -pad or None]


def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available


def chunk_it(image, tile_size):
    s = image.shape
    num_rows = math.ceil(s[0] / tile_size)
    num_cols = math.ceil(s[1] / tile_size)
    r = np.array_split(image, num_rows)
    rows = []
    for x in r:
        x = np.array_split(x, num_cols, axis=1)
        rows.append(x)
    return rows, num_rows, num_cols


def crop_center(img, pad):
    if pad == 0:
        return img
    return img[pad:-pad, pad: -pad, :]


def crop_flatten(chunked, pad):
    imgs = []
    for r, row in enumerate(chunked):
        for c, col in enumerate(row):
            col = crop_center(col, pad)
            imgs.append(col)
    return imgs


def patch_chips(imgs, n_rows, n_cols):
    h_stacks = []
    for i in range(n_rows):
        h_stacks.append(np.hstack(imgs[i * n_cols:n_cols * (i + 1)]))
    return np.vstack(h_stacks)


attribute_table = {
    'displayFieldName': '',
    'fieldAliases': {
        'OID': 'OID',
        'Value': 'Value',
        'Class': 'Class',
        'Red': 'Red',
        'Green': 'Green',
        'Blue': 'Blue'
    },
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Value',
            'type': 'esriFieldTypeInteger',
            'alias': 'Value'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Red',
            'type': 'esriFieldTypeInteger',
            'alias': 'Red'
        },
        {
            'name': 'Green',
            'type': 'esriFieldTypeInteger',
            'alias': 'Green'
        },
        {
            'name': 'Blue',
            'type': 'esriFieldTypeInteger',
            'alias': 'Blue'
        }
    ],
    'features': []
}


class ArcGISImageClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = 'Image Classifier'
        self.description = 'Image classification python raster function to inference a pytorch image classifier'
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        framework = self.json_info['Framework']
        # if 'ModelConfiguration' in self.json_info:
        #     if isinstance(self.json_info['ModelConfiguration'], str):
        #         ChildImageClassifier = getattr(importlib.import_module(
        #             '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildImageClassifier')
        #     else:
        #         ChildImageClassifier = getattr(importlib.import_module(
        #             '{}.{}'.format(framework, self.json_info['ModelConfiguration']['Name'])), 'ChildImageClassifier')
        # else:
        #     raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception(
                        "PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)

        self.child_image_classifier = ChildImageClassifier(model, model_as_file)
        self.n_target_classes = kwargs['n_target_classes']
        self.relu = nn.ReLU()
        self.proj_to_class_space = nn.Conv2d(
            self.child_image_classifier.model.layers[-1][-1].out_channels,
            self.n_target_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            device=torch.device("cuda") if torch.cuda.is_available() else torch.ddddddevice("cpu"),
        )
        # self.child_image_classifier = ChildImageClassifier()
        # self.child_image_classifier.initialize(model, model_as_file)

        # encoder = self.child_image_classifier.model.layers[:2]
        # self.encoder = nn.ModuleList([*encoder, self.child_image_classifier.model.layers[3][0]])
        self.encoder = nn.ModuleList([self.child_image_classifier.model.layers[0]])
        self.encoder.requires_grad_(False)

        self.encoder_head = nn.ModuleList([*self.child_image_classifier.model.layers[1:3], self.child_image_classifier.model.layers[3][0]])

        self.decoder = nn.ModuleList([self.child_image_classifier.model.layers[3][1], *self.child_image_classifier.model.layers[4:]])

    def getParameterInfo(self):
        required_parameters = [
            {
                'name': 'raster',
                'dataType': 'raster',
                'required': True,
                'displayName': 'Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]
        return self.child_image_classifier.getParameterInfo(required_parameters)

    def getConfiguration(self, **scalars):
        configuration = self.child_image_classifier.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        configuration['inheritProperties'] = 2 | 4 | 8
        configuration['inputMask'] = True
        return configuration

    def updateRasterInfo(self, **kwargs):
        kwargs['output_info']['bandCount'] = 1
        # todo: type is determined by the value range of classes in the json file
        prob_raster = getattr(self.child_image_classifier, 'probability_raster', False)
        if prob_raster:
            kwargs['output_info']['pixelType'] = 'f4'  # To ensure that output pixels are in prob range 0 to 1
        else:
            kwargs['output_info']['pixelType'] = 'i4'
        class_info = self.json_info['Classes']
        attribute_table['features'] = []
        for i, c in enumerate(class_info):
            attribute_table['features'].append(
                {
                    'attributes': {
                        'OID': i + 1,
                        'Value': c['Value'],
                        'Class': c['Name'],
                        'Red': c['Color'][0],
                        'Green': c['Color'][1],
                        'Blue': c['Color'][2]
                    }
                }
            )
        kwargs['output_info']['rasterAttributeTable'] = json.dumps(attribute_table)

        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # set pixel values in invalid areas to 0

        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        if self.json_info['ModelName'] == 'MultiTaskRoadExtractor':
            xx = self.child_image_classifier.detectRoads(tlc, shape, props, **pixelBlocks).astype(props['pixelType'],
                                                                                                  copy=False)
            pixelBlocks['output_pixels'] = xx
        elif hasattr(self.child_image_classifier, 'updatePixelsTTA'):
            xx = self.child_image_classifier.updatePixelsTTA(tlc, shape, props, **pixelBlocks).astype(
                props['pixelType'], copy=False)
            pixelBlocks['output_pixels'] = xx
        else:
            xx = self.child_image_classifier.updatePixels(tlc, shape, props, **pixelBlocks).astype(props['pixelType'],
                                                                                                   copy=False)
            tytx = getattr(self.child_image_classifier, 'tytx', self.json_info['ImageHeight'])
            chunks, num_rows, num_cols = chunk_it(xx.transpose(1, 2, 0),
                                                  tytx)  # self.json_info['ImageHeight'])  # ImageHeight = ImageWidth
            xx = patch_chips(crop_flatten(chunks, self.child_image_classifier.padding), num_rows, num_cols)
            xx = xx.transpose(2, 0, 1)
            pixelBlocks['output_pixels'] = xx

        return pixelBlocks

    def preprocess_image(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if len(x.shape) == 4:
            if x.shape[1] == 13:
                x = x[:, :12]
        else:
            raise RuntimeError(f"Input is not batched: '{x.shape}'")

        if normalize:
            normalization_stats = self.child_image_classifier.json_info['NormalizationStats']
            normalize_fun = transforms.Normalize(
                normalization_stats['band_mean_values'], normalization_stats['band_std_values']
            )
            x = normalize_fun(x / 255.0)
        return x

    def forward(self, x: torch.Tensor, normalize: bool = True, only_encoding: bool = False):
        x = self.preprocess_image(x, normalize=normalize)
        if only_encoding:
            for layer in self.encoder:
                x = layer(x)
            return x

        return self.proj_to_class_space(self.relu(self.child_image_classifier.model(x)))

    def pixel_wise_features(self, img: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self(img, normalize=normalize, only_encoding=True)

    def run_encoder_head(self, pixel_wise_features: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder_head:
            pixel_wise_features = layer(pixel_wise_features)

        return pixel_wise_features

    def run_decoder(self, embedded_features: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            embedded_features = layer(embedded_features)

        return embedded_features

    # def to(self, device: torch.device):
    #     self.child_image_classifier.model.to(device)
    #     self.proj_to_class_space = self.proj_to_class_space.to(device)

    def parameters(self, **kwargs):
        return self.child_image_classifier.model.parameters()
