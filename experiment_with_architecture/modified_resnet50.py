import torch
import torch.nn as nn
import torchvision.models as models

class NaiveInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NaiveInception, self).__init__()

        # Allocate filters to each branch so the total after concatenation is out_channels (512)
        b1_channels = out_channels // 4  # 1x1 conv branch: 128
        b2_channels = out_channels // 4  # 3x3 conv branch: 128
        b3_channels = out_channels // 4  # 5x5 conv branch: 128
        b4_channels = out_channels - b1_channels - b2_channels - b3_channels  # pool branch: 128

        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, b1_channels, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )

        # 3x3 convolution branch with 'same' padding
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, b2_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )

        # 5x5 convolution branch with 'same' padding
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, b3_channels, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True)
        )

        # 3x3 max pooling branch followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 'same' padding
            nn.Conv2d(in_channels, b4_channels, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)

        # Concatenate the outputs along the channel dimension
        output = torch.cat([branch1_output, branch2_output, branch3_output, branch4_output], dim=1)
        return output


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModifiedResNet50, self).__init__()

        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # ResNet50 layers up to conv3_block4_out
        self.initial_layers = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )

        self.layer1 = resnet50.layer1  # conv2_x
        self.layer2 = resnet50.layer2  # conv3_x, which includes conv3_block4 at the end
        # This ends with output shape (17, 17, 512)

        # New convolutional layer after conv3_block4_out
        self.new_conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # Naive inception layer
        self.inception = NaiveInception(in_channels=512, out_channels=512)

        # Final convolutional layer
        self.new_conv2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Finish with global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        # Freeze conv2 layers and before
        self._freeze_layers()

    def _freeze_layers(self):
        # Freeze initial layers and layer1 (conv2_x and earlier)
        for param in self.initial_layers.parameters():
            param.requires_grad = False

        for param in self.layer1.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Forward pass through frozen layers
        x = self.initial_layers(x)
        x = self.layer1(x)

        # Forward pass through layer2 (conv3_x)
        x = self.layer2(x)

        # Forward pass through new layers
        x = self.new_conv1(x)
        x = self.inception(x)
        x = self.new_conv2(x)

        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Example usage
if __name__ == "__main__":
    model = ModifiedResNet50(num_classes=3)
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [1, 1000]

    # To check the feature map size at each layer:
    x = dummy_input
    x = model.initial_layers(x)
    print(f"After initial layers: {x.shape}")
    x = model.layer1(x)
    print(f"After layer1: {x.shape}")
    x = model.layer2(x)
    print(f"After layer2 (conv3_block4_out): {x.shape}")
    x = model.new_conv1(x)
    print(f"After new_conv1: {x.shape}")
    x = model.inception(x)
    print(f"After inception: {x.shape}")
    x = model.new_conv2(x)
    print(f"After new_conv2: {x.shape}")
