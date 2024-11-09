from torchvision.models import resnet50
from pathlib import Path
import torch
import torch.nn as nn
from Project.top_level.utils.sen12ms_to_clc import CLCValues
import numpy as np

ModelsPath: Path = Path(__file__).parent.resolve() / "initmodel"

# SimplifiedClassMap: dict[int, int] = {
#     0: CLCValues.ForestAndSemiNaturalAreas.value,
#     1: CLCValues.ForestAndSemiNaturalAreas.value,
#     2: CLCValues.ForestAndSemiNaturalAreas.value,
#     3: CLCValues.Wetlands.value,
#     4: CLCValues.AgriculturalAreas.value,
#     5: CLCValues.ArtificialSurfaces,
#     6: CLCValues.ForestAndSemiNaturalAreas,
#     7: CLCValues.WaterBodies,
#     8: CLCValues.NoData,
# }

SimplifiedClassMap: torch.Tensor = torch.Tensor([
    CLCValues.ForestAndSemiNaturalAreas.value,
    CLCValues.ForestAndSemiNaturalAreas.value,
    CLCValues.ForestAndSemiNaturalAreas.value,
    CLCValues.Wetlands.value,
    CLCValues.AgriculturalAreas.value,
    CLCValues.ArtificialSurfaces.value,
    CLCValues.ForestAndSemiNaturalAreas.value,
    CLCValues.WaterBodies.value,
    CLCValues.NoData.value,
])


class HsgAimlResnet50(nn.Module):
    def __init__(self, n_simplified_igbp_classes: int = 8):
        super().__init__()
        self.backbone2 = resnet50(weights=None, num_classes=n_simplified_igbp_classes)
        dim_mlp2 = self.backbone2.fc.in_features

        self.backbone_seq = nn.Sequential(
            self.backbone2.bn1,
            self.backbone2.relu,
            self.backbone2.maxpool,
            self.backbone2.layer1,
            self.backbone2.layer2,
            self.backbone2.layer3,
            self.backbone2.layer4,
        )
        self.fc = torch.nn.Linear(dim_mlp2, n_simplified_igbp_classes, bias=True)
        self.backbone2.fc = torch.nn.Identity()

    def forward(self, x):
        # x2 = self.backbone2(x)
        x2 = self.backbone_seq(self.backbone2.conv1(x))
        x2 = self.backbone2.fc(self.backbone2.avgpool(x2))

        z = self.fc(x2)

        return SimplifiedClassMap[z]

    def load_trained_state_dict(self, weights, freeze: bool = False):
        for k in list(weights.keys()):
            if k.startswith(('backbone1.fc', 'backbone2.fc')):
                del weights[k]

        log = self.load_state_dict(weights, strict=False)
        # assert log.missing_keys == ['fc.weight', 'fc.bias']
        assert [x for x in log.missing_keys if not x.startswith("backbone_seq")] == ["fc.weight", "fc.bias"]

        if freeze:
            for name, param in self.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False


def load_resnet50_hsg_aiml():
    in_channels = 13
    model = HsgAimlResnet50()
    model.backbone2.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    model_path = ModelsPath / "hsg-aiml--resnet50.pth"
    state_dict = torch.load(str(model_path), weights_only=True)['state_dict']
    model.load_trained_state_dict(state_dict)

    return model


def main():
    model = load_resnet50_hsg_aiml()


if __name__ == '__main__':
    main()
