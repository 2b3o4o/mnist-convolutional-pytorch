import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()

        modules = [nn.Dropout(p=dropout_rate)] if dropout_rate > 0 else []
        modules.extend([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ])
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, relu):
        super().__init__()

        modules = [nn.Dropout(p=dropout_rate)] if dropout_rate > 0 else []
        modules.extend([
            nn.Linear(in_channels, out_channels),
        ])
        if relu:
            modules.append(nn.ReLU())
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        dropout_rate = 0.5

        self.conv = nn.ModuleList([
            ConvLayer(in_channels=1, out_channels=16, dropout_rate=0),
            ConvLayer(in_channels=16, out_channels=64, dropout_rate=dropout_rate),
            ConvLayer(in_channels=64, out_channels=64, dropout_rate=dropout_rate)
        ])

        self.flatten = nn.Flatten()

        self.linear = nn.ModuleList([
            LinearLayer(in_channels=1024, out_channels=hidden_size, dropout_rate=dropout_rate, relu=True),
            LinearLayer(in_channels=hidden_size, out_channels=hidden_size, dropout_rate=dropout_rate, relu=True),
            LinearLayer(in_channels=hidden_size, out_channels=output_size, dropout_rate=dropout_rate, relu=False)
        ])

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.linear:
            x = layer(x)
        return x