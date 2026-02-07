import torch
import torch.nn as nn
from torchvision import models


class InceptionV3_Final(models.Inception3): # 继承原模型类
    def __init__(self, num_classes=2):
        # 调用父类构造函数，关闭辅助分支
        super(InceptionV3_Final, self).__init__(num_classes=num_classes, aux_logits=False, init_weights=True)

        # 这里的命名必须与你训练时保存的键名完全一致
        # 如果你训练时修改的是 self.fc，这里就直接修改 self.fc
        num_ftrs = self.fc.in_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.6),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 覆盖父类的 forward，简化逻辑（因为已经关闭了 aux_logits）
        return super(InceptionV3_Final, self).forward(x)


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=output_size)
        )

    def forward(self, x):
        return self.classifier(x)