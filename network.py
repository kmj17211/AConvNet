import torch.nn as nn

class AConvnet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AConvnet, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5, stride = 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(16, 32, kernel_size = 5, stride = 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(32, 64, kernel_size = 6, stride = 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(64, 128, kernel_size = 5, stride = 1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            
            nn.Conv2d(128, num_classes, kernel_size = 3, stride = 1),
        )

        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.convnet(x)
        x = self.flat(x)
        return x