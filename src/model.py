import torch.nn as nn

class ConvLayer(nn.module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        modules = [nn.Dropout(p=dropout_rate)] if dropout_rate > 0 else []
        modules.extend([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ])
        self.layers = nn.ModuleList([modules])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LinearLayer(nn.module):
    def __init__(self, in_channels, out_channels, dropout_rate, relu):
        modules = [nn.Dropout(p=dropout_rate)] if dropout_rate > 0 else []
        modules.extend([
            nn.Linear(in_channels=in_channels, out_channels=out_channels),
        ])
        if relu:
            modules.append(nn.ReLU())
        self.layers = nn.ModuleList([modules])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        dropout_rate = 0.5

        self.conv = nn.ModuleList([
            ConvLayer(in_channels=1, out_channels=16, dropout_rate=0),
            ConvLayer(in_channels=16, out_channels=64, dropout_rate=dropout_rate),
            ConvLayer(in_channels=64, out_channels=64, dropout_rate=dropout_rate)
        ])

        self.flatten = nn.Flatten()

        self.linear = nn.ModuleList([
            LinearLayer(in_channels=(26 // 2 // 2 + 1) ** 2 * 64, out_channels=64, dropout_rate=dropout_rate, relu=True),
            LinearLayer(in_channels=64, out_channels=64, dropout_rate=dropout_rate, relu=True),
            LinearLayer(in_channels=64, out_channels=output_size, dropout_rate=dropout_rate, relu=False)
        ])

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.linear:
            x = layer(x)
        return x