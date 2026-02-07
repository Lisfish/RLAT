import torch
import torch.nn as nn
from torchvision import models

class VGG16_Final(nn.Module):
    def __init__(self, num_classes=2): # 确保对应你的二分类
        super().__init__()
        self.base = models.vgg16_bn(weights=None) # 无需下载预训练权重，我们将加载本地pth
        self.base.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(0.6),
            nn.Linear(4096, 256),
            nn.ReLU(True),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
victim_model = VGG16_Final(num_classes=2).to(device)
victim_model.load_state_dict(torch.load("best_vgg16.pth", map_location=device, weights_only=True))
victim_model.eval() # 必须设为 eval 模式

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear( in_features= input_size, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=output_size)
        )
    def forward(self, x):
        return self.classifier(x)