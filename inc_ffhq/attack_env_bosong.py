import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from my_models import InceptionV3_Final  # 确保导入新类


class AttackEnv:
    def __init__(self, model_path, data_dir, img_size=299, mask_size=23):  # 299/23 ≈ 13格
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.mask_size = mask_size
        self.num_grids = img_size // mask_size

        # 1. 加载 InceptionV3
        self.model = InceptionV3_Final(num_classes=2).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True),
            strict=False
        )
        self.model.eval()

        # 2. 预处理：使用 ImageNet 的归一化参数
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.dataset = datasets.ImageFolder(data_dir, transform=self.transform)
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
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
            target_area = self.current_img[:, :, x_min:x_min + self.mask_size, y_min:y_min + self.mask_size]

            # --- 修复点：增加 relu() 或 clamp() 确保非负 ---
            # 即使 target_area + 2.1 出现了极小的负数，也能被拉回 0
            shifted_area = torch.clamp((target_area + 2.1) * 10.0, min=0.0)

            # 现在可以安全生成泊松噪声
            poisson_sample = torch.poisson(shifted_area) / 10.0 - 2.1

            # 3. 计算扰动并应用
            noise = (poisson_sample - target_area) * self.noise_scale
            self.current_img[:, :, x_min:x_min + self.mask_size, y_min:y_min + self.mask_size] += noise

            # 4. 限制范围
            self.current_img = torch.clamp(self.current_img, -2.1, 2.7)

            output = self.model(self.current_img)
            new_pred = torch.argmax(output, dim=1).item()

        l2_dist = torch.norm(self.current_img - self.original_img).item()
        done = (new_pred != self.label.item())
        reward = 10.0 - l2_dist if done else -0.1 - (l2_dist * 0.01)

        return self.get_state(), reward, done, {"l2": l2_dist, "pred": new_pred}

    def get_state(self):
        with torch.no_grad():
            # InceptionV3 提取全局池化前的特征
            # 输入 299x299 -> 特征图通常是 8x8x2048
            x = self.model.Conv2d_1a_3x3(self.current_img)
            x = self.model.Conv2d_2a_3x3(x)
            x = self.model.Conv2d_2b_3x3(x)
            x = self.model.maxpool1(x)
            x = self.model.Conv2d_3b_1x1(x)
            x = self.model.Conv2d_4a_3x3(x)
            x = self.model.maxpool2(x)
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
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return x.squeeze(0)