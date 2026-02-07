import torch
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from attack_env_bosong import AttackEnv
from rl_agent import DQNAgent
from datetime import datetime
# ==========================================
# 1. 全局配置
# ==========================================
CONFIG = {
    "MODEL_PATH": "best.pth", # 指向你训练好的 InceptionV3 权重
    "DATA_DIR": "./data",
    "IMG_SIZE": 32,
    "MASK_SIZE": 4,
    "EPSILON": 1.0,
    "EPS_DECAY": 0.995,
    "EPS_MIN": 0.1,
    "GAMMA": 0.9,
    "BATCH_SIZE": 32,
    "TARGET_UPDATE": 10,
    "MAX_EPISODES": 500,
    "MAX_STEPS": 30
}

def denormalize_and_save(img_tensor, save_path):
    # 使用 CIFAR-100 的统计值
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    save_image(img_tensor, save_path)


# ==========================================
# 3. 主训练与评估函数
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建保存路径
    img_root_dir = "attack_results"
    model_root_dir = "dqn_models"
    img_save_dir = os.path.join(img_root_dir, f"run_{timestamp}")
    model_save_dir = os.path.join(model_root_dir, f"run_{timestamp}")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # 初始化环境
    env = AttackEnv(
        model_path=CONFIG["MODEL_PATH"],
        data_dir=CONFIG["DATA_DIR"],
        img_size=CONFIG["IMG_SIZE"],
        mask_size=CONFIG["MASK_SIZE"]
    )
    initial_state = env.reset()
    input_size = initial_state.shape[0]  # 对于 InceptionV3 + GAP 是 2048
    output_size = (CONFIG["IMG_SIZE"] // CONFIG["MASK_SIZE"]) ** 2
    class_names = env.classes
    print(f"类别映射已识别: {env.dataset.class_to_idx}")

    agent = DQNAgent(input_size, output_size, CONFIG)

    # 统计变量
    original_correct = 0  # 原始模型预测正确的数量
    success_count = 0  # 攻击成功次数
    total_episodes = CONFIG["MAX_EPISODES"]

    print("\n" + "=" * 50)
    print("开始强化学习攻击训练与预测对比...")
    print("=" * 50 + "\n")

    for episode in range(total_episodes):
        state = env.reset()

        # 获取基础标签信息
        true_label = env.label.item()
        # 在初始状态下，环境已经记录了初始预测
        # 注意：这里我们直接从环境里拿第一次评估的结果
        with torch.no_grad():
            initial_output = env.model(env.current_img)
            init_pred = torch.argmax(initial_output, dim=1).item()
        top5_init = torch.topk(initial_output, k=5, dim=1).indices.squeeze(0)

        # 统计原始准确率
        if true_label in top5_init.tolist():
            original_correct += 1

        total_reward = 0
        success = False
        final_pred = init_pred

        for step in range(CONFIG["MAX_STEPS"]):
            # Agent 选择动作
            action = agent.select_action(state, mode="train")

            # 环境执行动作
            next_state, reward, done, info = env.step(action)

            # 记录最新的预测值
            final_pred = info["pred"]

            # 存储与学习
            agent.store_transition(state, action, next_state, reward)
            agent.optimize_model()

            state = next_state
            total_reward += reward

            if done:
                success = True
                break

        if success:
            success_count += 1

        # 打印当前样本的详细对比
        status_str = "成功" if success else "失败"
        print(f"[{episode + 1:03d}/{total_episodes}] "
              f"真实标签: {class_names[true_label]:<5} | "
              f"原始预测: {class_names[init_pred]:<5} | "
              f"攻击后预测: {class_names[final_pred]:<5} | "
              f"Top5中是否包含GT: {true_label in info['top5']} |"
              f"结果: {status_str}")

        # 定期保存图片 (仅保存前50个或成功案例以节省空间)
        if episode < 50 or success:
            img_name = f"ep{episode + 1}_{class_names[true_label]}_to_{class_names[final_pred]}.png"
            denormalize_and_save(env.current_img.squeeze(0).cpu(), os.path.join(img_save_dir, img_name))

        # 定期更新目标网络
        if episode % CONFIG["TARGET_UPDATE"] == 0:
            agent.update_target_network()

        # 每 50 轮保存一次模型
        if (episode + 1) % 50 == 0:
            model_path = os.path.join(model_save_dir, f"dqn_ep{episode + 1}.pth")
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"模型已保存至: {model_path}")
    # ==========================================
    # 4. 最终结果汇总
    # ==========================================
    orig_acc = (original_correct / total_episodes) * 100
    attack_sr = (success_count / total_episodes) * 100

    print("\n" + "=" * 50)
    print(f"训练评估完成！")
    print(f"原始模型 Top-5 准确率 (Original Top-5 Accuracy): {orig_acc:.2f}%")
    print(f"攻击成功率 (Attack Success Rate): {attack_sr:.2f}%")
    print(f"本轮结果图片路径: {os.path.abspath(img_save_dir)}")
    print(f"本轮模型文件路径: {os.path.abspath(model_save_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    main()