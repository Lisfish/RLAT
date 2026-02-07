import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. 配置路径与参数 ---
train_dir = "E:/dataset/small_data/train"
val_dir = "E:/dataset/small_data/val"
test_dir ="E:/dataset/small_data/test"
save_path = "E:/代码/best_model.pth"


batch_size = 32
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据增强与加载 ---
transform_train = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)
class_names = train_dataset.classes


# --- 3. 构建模型 (InceptionV3) ---
def build_model(num_classes):
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

    # 冻结部分层 (模拟原代码逻辑)
    # InceptionV3 在 torchvision 中层级较深，这里冻结前 2/3 的特征提取层
    children = list(model.children())
    for child in children[:15]:
        for param in child.parameters():
            param.requires_grad = False

    model.aux_logits = False  # 关闭辅助分支

    # 修改全连接层以匹配你的分类数
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Dropout(0.6),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.6),
        nn.Linear(256, num_classes)
    )
    return model.to(device)


model = build_model(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.002)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

# --- 4. 训练与验证循环 ---
best_acc = 0.0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss, train_corrects = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_corrects += torch.sum(preds == labels.data)

    epoch_loss = train_loss / len(train_dataset)
    epoch_acc = train_corrects.double() / len(train_dataset)

    # 验证阶段
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)

    v_loss = val_loss / len(val_dataset)
    v_acc = val_corrects.double() / len(val_dataset)

    scheduler.step(v_loss)

    print(
        f'Epoch {epoch}/{epochs - 1} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}')

    # 保存最佳权重
    if v_acc > best_acc:
        best_acc = v_acc
        torch.save(model.state_dict(), save_path)

# --- 5. 测试集最终评估与预测 ---
print("\n--- 开始测试集评估 ---")
model.load_state_dict(torch.load(save_path, weights_only=True))  # 加载最优权重
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 输出分类报告
print(classification_report(all_labels, all_preds, target_names=class_names))

# 可视化前5个测试样本
'''plt.figure(figsize=(15, 5))
for i in range(5):
    img = test_dataset[i][0].permute(1, 2, 0).numpy()
    img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)  # 反归一化
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(f"P: {class_names[all_preds[i]]}\nT: {class_names[all_labels[i]]}")
    plt.axis('off')
plt.show()'''