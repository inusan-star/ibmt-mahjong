import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """
    ResNet Block
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        )
        self.bn2 = nn.BatchNorm2d(dim)

        self.apply(self._init_weights)

    def _init_weights(
        self,
        module,
        weight_init_val=1.0,
        bias_init_val=0.0,
    ):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

            if module.bias is not None:
                nn.init.constant_(module.bias, bias_init_val)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, weight_init_val)
            nn.init.constant_(module.bias, bias_init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(identity + out)
        return out


class CNN(nn.Module):
    """
    CNN
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        channels: int,
        num_classes: int,
        num_block: int,
        hidden_dim: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ):
        super().__init__()

        self.image_height, self.image_width = image_size
        self.channels = channels
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resnet_blocks = nn.Sequential(
            *[
                ResNetBlock(
                    dim=hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
                for _ in range(num_block)
            ]
        )
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=(1, 1), bias=True)

        self.apply(self._init_weights)

    def _init_weights(
        self,
        module,
        weight_init_val=1.0,
        bias_init_val=0.0,
    ):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

            if module.bias is not None:
                nn.init.constant_(module.bias, bias_init_val)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, weight_init_val)
            nn.init.constant_(module.bias, bias_init_val)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass
        """
        _, channels, height, width = x.shape

        assert channels == self.channels, f"Expected channels {self.channels}, got {channels}"
        assert height == self.image_height, f"Expected height {self.image_height}, got {height}"
        assert width == self.image_width, f"Expected width {self.image_width}, got {width}"

        x = self.conv1(x)
        x = self.resnet_blocks(x)
        x = self.conv2(x)

        action = x.squeeze(1).flatten(1)

        assert action.shape[1] == self.num_classes, f"Expected classes {self.num_classes}, got {action.shape[1]}"

        return {"action": action}
