import torch
import torchvision
from torch import nn
from torchvision.models import get_model, get_model_weights


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, in_channels, view_per_image, out_channels, pretrained=False,
                 process_views_separately=False, add_mid_view=False, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.model_name = model_name
        self.add_mid_view = add_mid_view
        self.view_per_image = view_per_image
        self.process_views_separately = process_views_separately

        self.in_channels = in_channels if process_views_separately else in_channels * view_per_image
        self.out_channels = out_channels if process_views_separately else out_channels * view_per_image

        assert "resnet" in model_name, "Only resnet is supported for now"

        kwargs = {}
        if pretrained:
            kwargs['weights'] = get_model_weights(model_name).IMAGENET1K_V1
        self.net = get_model(model_name, **kwargs)

        # Replace first conv layer
        conv = self.net.conv1
        assert isinstance(conv, nn.Conv2d)
        new_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, bias=conv.bias is not None,
            padding_mode=conv.padding_mode
        )
        self.net.conv1 = new_conv

        # Add additional projection later at the end
        self.outfc = nn.Linear(self.net.fc.out_features, self.out_channels)

    def forward(self, x):
        Nimg, Nview, C, H, W = x.shape
        if self.add_mid_view:
            assert Nview == self.view_per_image - 1 and Nview % 2 == 0
            x = torch.cat([x[:, :Nview//2], torch.zeros_like(x[:, [0]]), x[:, Nview//2:]], dim=1)
            Nview += 1

        if self.process_views_separately:
            x = x.view(Nimg * Nview, C, H, W)
        else:
            x = x.view(Nimg, Nview * C, H, W)

        x = self.net(x)
        x = self.outfc(x)  # [Nimg * Nview, out_channels] or [Nimg, Nview * out_channels]
        x = x.view(Nimg, Nview, -1)

        return x
