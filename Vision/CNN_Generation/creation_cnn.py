import torch
import torch.nn as nn

"""MAKE SURE THIS IS THE RIGHT PATH FOR YOU"""
path = "Vision/CNN_Generator/team_air_gcp_localization_protoype.pth"

class ConvNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Input 94 x 80 x 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=0),  # 92 x 78 x 8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)       # 46 x 39 x 8
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, groups=8, padding=0), # 44 x 37 x 8
            nn.Conv2d(8, 12, kernel_size=1, bias=False), # Depthwise separable convolution
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)               # 22 x 18 x 12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, groups=12, padding=0), # 20 x 16 x 12
            nn.Conv2d(12, 16, kernel_size=1, bias=False), # Depthwise separable convolution
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                   # 10 x 8 x 16
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, groups=16, padding=0), # 8 x 6 x 16
            nn.Conv2d(16, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU()                                               # 8 x 6 x 24
        )

        # gap might be a bad idea
        # but maybe a smaller adaptive pooling layer will work better
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )

  def forward(self, x):
        # Input in (Batch, Channels, Height, Width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

model = ConvNet()
model.load_state_dict(torch.load(path, weights_only=True))
model = model.to(device)
model.eval()
exported_program = torch.export.export(model, args=(torch.randn(1, 1, 94, 80).to(device),))
torch.export.save(exported_program, "team_air_gcp_localization_protoype.pt2")