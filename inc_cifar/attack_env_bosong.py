import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from my_models import InceptionV3
from torchvision import datasets


class AttackEnv:
    def __init__(self, model_path, data_dir, img_size=224, mask_size=16): # VGG通常用224
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.mask_size = mask_size
        self.num_grids = img_size // mask_size

        # 1. 加载 VGG-16
        self.model = InceptionV3().to(self.device)
        # 加载权重
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # 2. 预处理：VGG-16 常用 224x224 分辨率
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])

        self.dataset = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        # 获取类别名称
        self.classes = self.dataset.classes
        self.data_iter = iter(self.data_loader)
        self.noise_scale = 0.1

    def reset(self):
        try:
            self.current_img, self.label = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            self.current_img, self.label = next(self.data_iter)

        self.current_img = self.current_img.to(self.device)
        self.label = self.label.to(self.device)
        self.original_img = self.current_img.clone()
        return self.get_state()

    def step(self, action):
        i = action // self.num_grids
        j = action % self.num_grids
        x_min, y_min = i * self.mask_size, j * self.mask_size

        with torch.no_grad():
            # 1. 提取选定的局部区域
            target_area = self.current_img[:, :, x_min:x_min + self.mask_size, y_min:y_min + self.mask_size]

            # 2. 生成泊松噪声
            # 注意：泊松分布的输入（lambda）必须为非负。
            # 由于图像经过了归一化（包含负数），我们需要先将其平移到非负区间
            # 或者假设一个基准亮度值
            shifted_area = (target_area + 2.1) * 10.0  # 示例：平移并放大以获得明显的采样特征

            # 使用 torch.poisson 生成噪声
            poisson_sample = torch.poisson(shifted_area) / 10.0 - 2.1

            # 3. 计算扰动并应用
            noise = (poisson_sample - target_area) * self.noise_scale
            self.current_img[:, :, x_min:x_min + self.mask_size, y_min:y_min + self.mask_size] += noise

            # 4. 限制范围
            self.current_img = torch.clamp(self.current_img, -2.1, 2.7)

            output = self.model(self.current_img)

            # Top-1 预测（用于日志）
            new_pred = torch.argmax(output, dim=1).item()

            # Top-5 预测（用于攻击判定）
            top5_preds = torch.topk(output, k=5, dim=1).indices.squeeze(0)
            true_label = self.label.item()

        l2_dist = torch.norm(self.current_img - self.original_img).item()
        done = (true_label not in top5_preds.tolist())
        reward = 10.0 - l2_dist if done else -0.1 - (l2_dist * 0.01)

        return self.get_state(), reward, done, {
            "l2": l2_dist,
            "pred": new_pred,
            "top5": top5_preds.tolist()
        }

    def get_state(self):
        """
        使用 InceptionV3 的倒数第二层特征作为状态
        输出维度: 2048
        """
        with torch.no_grad():
            x = self.current_img

            # ===== 手动走到 linear 前一层 =====
            x = self.model.Conv2d_1a_3x3(x)
            x = self.model.Conv2d_2a_3x3(x)
            x = self.model.Conv2d_2b_3x3(x)
            x = self.model.Conv2d_3b_1x1(x)
            x = self.model.Conv2d_4a_3x3(x)

            x = self.model.Mixed_5b(x)
            x = self.model.Mixed_5c(x)
            x = self.model.Mixed_5d(x)

            x = self.model.Mixed_6a(x)
            x = self.model.Mixed_6b(x)
            x = self.model.Mixed_6c(x)
            x = self.model.Mixed_6d(x)
            x = self.model.Mixed_6e(x)

            x = self.model.Mixed_7a(x)
            x = self.model.Mixed_7b(x)
            x = self.model.Mixed_7c(x)

            x = self.model.avgpool(x)  # (B, 2048, 1, 1)
            x = x.view(x.size(0), -1)  # (B, 2048)

            return x.squeeze(0)
