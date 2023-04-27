from typing import Tuple, List
import numpy as np
from mindspore import Tensor
from mindspore import nn, ops
import mindspore.common.initializer as init
from .mindcv_models.convnext import ConvNextLayerNorm, Block, default_cfgs
from .mindcv_models.utils import load_pretrained
from ._registry import register_backbone, register_backbone_class

__all__ = ['DetConvNeXt', 'det_convnext_tiny', 'det_convnext_small']


@register_backbone_class
class DetConvNeXt(nn.Cell):
    r"""ConvNeXt model class, based on
    '"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>'
    Args:
        in_channels (int) : dim of the input channel.
        num_classes (int) : dim of the classes predicted.
        depths (List[int]) : the depths of each layer.
        dims (List[int]) : the middle dim of each layer.
        drop_path_rate (float) : the rate of droppath default : 0.
        layer_scale_init_value (float) : the parameter of init for the classifier default : 1e-6.
        head_init_scale (float) : the parameter of init for the head default : 1.
    """

    def __init__(
        self,
        depths: List[int],
        dims: List[int],
        in_channels: int = 3,
        num_classes: int = 1000,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()

        self.downsample_layers = nn.CellList()  # stem and 3 intermediate down_sampling conv layers
        stem = nn.SequentialCell(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, has_bias=True),
            ConvNextLayerNorm((dims[0],), epsilon=1e-6, norm_axis=1),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                ConvNextLayerNorm((dims[i],), epsilon=1e-6, norm_axis=1),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, has_bias=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.CellList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = list(np.linspace(0, drop_path_rate, sum(depths)))
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value))
            stage = nn.SequentialCell(blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = ConvNextLayerNorm((dims[-1],), epsilon=1e-6)  # final norm layer

        self.f0 = self.downsample_layers[0]
        self.f1 = self.stages[0]
        self.f2 = self.downsample_layers[1]
        self.f3 = self.stages[1]
        self.f4 = self.downsample_layers[2]
        self.f5 = self.stages[2]
        self.f6 = self.downsample_layers[3]
        self.f7 = self.stages[3]
        self.head_init_scale = head_init_scale
        self._initialize_weights()

        self.out_channels = dims


    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(
                    init.initializer(init.TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype)
                )
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))


    def construct(self, x: Tensor) -> Tensor:
        x = self.f0(x)
        x1 = self.f1(x)

        x2 = self.f2(x1)
        x2 = self.f3(x2)

        x3 = self.f4(x2)
        x3 = self.f5(x3)

        x4 = self.f6(x3)
        x4 = self.f7(x4)

        return [x1, x2, x3, x4]


@register_backbone
def det_convnext_tiny(pretrained: bool = False, **kwargs):
    """Get ConvNeXt tiny model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_tiny"]
    model = DetConvNeXt(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, auto_mapping=True)

    return model


@register_backbone
def det_convnext_small(pretrained: bool = False, **kwargs):
    """Get ConvNeXt small model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_small"]
    model = DetConvNeXt(
        depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, auto_mapping=True)

    return model
