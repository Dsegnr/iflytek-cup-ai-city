# train_q_learning.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime


# ----------------------Matplotlib中文字体配置----------------------
def setup_matplotlib_chinese():
    """配置Matplotlib支持中文显示，避免乱码"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("[初始化] Matplotlib中文字体配置完成")
    except Exception as e:
        print(f"[初始化] 中文字体配置警告：{str(e)}，将使用默认字体")


# 初始化中文配置
setup_matplotlib_chinese()

# ----------------------模型配置----------------------
MODEL_CONFIG = {
    "congestion_threshold": 0.7,
    "prediction_horizon": 10,
    "q_learning": {
        "episodes": 600,
        "alpha": 0.15,
        "gamma": 0.9,
        "epsilon": 0.1
    },
    "real_data_config": {
        "data_path": "real_lane_traffic_data.csv",
        "seq_len": 8
    }
}


# ----------------------交通路口环境----------------------
class TrafficIntersectionEnv:
    def __init__(self):
        self.ns_queue = np.random.randint(5, 15)
        self.ew_queue = np.random.randint(5, 15)
        self.max_queue = 30
        self.time_step = 0
        self.max_steps = 100

    def reset(self):
        self.ns_queue = np.random.randint(5, 15)
        self.ew_queue = np.random.randint(5, 15)
        self.time_step = 0
        return self._get_state()

    def _get_state(self):
        ns_level = 0 if self.ns_queue < 10 else 1 if self.ns_queue < 20 else 2
        ew_level = 0 if self.ew_queue < 10 else 1 if self.ew_queue < 20 else 2
        return ns_level * 3 + ew_level

    def step(self, action):
        self.time_step += 1
        reward = 0

        if action == 0:  # 南北绿灯
            self.ns_queue = max(0, self.ns_queue - 5)
            self.ew_queue = min(self.max_queue, self.ew_queue + 3)
            reward = (5 - 3) if self.ns_queue > 0 else -3
        else:  # 东西绿灯
            self.ew_queue = max(0, self.ew_queue - 5)
            self.ns_queue = min(self.max_queue, self.ns_queue + 3)
            reward = (5 - 3) if self.ew_queue > 0 else -3

        done = True if self.time_step >= self.max_steps else False
        return self._get_state(), reward, done


# ----------------------Q-learning智能体----------------------
class QLearningAgent:
    def __init__(self, n_states=9, n_actions=2, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state, :])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q


# ----------------------Q-learning模型训练模块----------------------
class QLearningTrainer:
    def __init__(self, config):
        self.config = config
        self.model_path = "trained_q_learning_model.pkl"

    def train(self, training_data=None):
        """训练Q-learning模型并保存"""
        print("🚦 开始训练Q-learning模型...")
        print(f"📊 训练参数: {self.config['q_learning']['episodes']}轮, alpha={self.config['q_learning']['alpha']}")

        env = TrafficIntersectionEnv()
        agent = QLearningAgent(
            n_states=9,
            n_actions=2,
            alpha=self.config["q_learning"]["alpha"],
            gamma=self.config["q_learning"]["gamma"],
            epsilon=self.config["q_learning"]["epsilon"]
        )

        total_rewards = self._train_agent(agent, env, self.config["q_learning"]["episodes"])

        # 保存训练好的模型
        self._save_model(agent)

        # 绘制训练曲线
        self._plot_training_curve(total_rewards)

        # 显示训练结果
        self._display_training_results(agent, total_rewards)

        return agent

    def _train_agent(self, agent, env, episodes):
        """训练智能体"""
        total_rewards = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.update_q_table(state, action, reward, next_state)
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                print(f"训练轮次：{episode + 1}/{episodes}，本轮总奖励：{total_reward:.2f}")

        return total_rewards

    def _save_model(self, agent):
        """保存训练好的模型"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(agent, f)
            print(f"✅ Q-learning模型已保存到: {self.model_path}")

            # 验证文件是否成功创建
            if os.path.exists(self.model_path):
                file_size = os.path.getsize(self.model_path)
                print(f"✅ 模型文件验证成功，文件大小: {file_size} 字节")
            else:
                print("❌ 模型文件保存失败")

        except Exception as e:
            print(f"❌ 模型保存失败: {str(e)}")

    def _plot_training_curve(self, total_rewards):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(total_rewards)
        plt.xlabel("训练轮次")
        plt.ylabel("每轮总奖励")
        plt.title("Q-Learning智能体训练曲线")
        plt.grid(True)
        plt.savefig("q_learning_training_curve.png")
        plt.close()
        print("📈 训练曲线已保存为: q_learning_training_curve.png")

    def _display_training_results(self, agent, total_rewards):
        """显示训练结果"""
        print("\\n" + "=" * 50)
        print("🎉 训练完成！")
        print("=" * 50)
        print(f"📊 最终Q-table形状: {agent.q_table.shape}")
        print(f"📈 平均奖励: {np.mean(total_rewards[-100:]):.2f}")  # 最后100轮的平均奖励
        print(f"📈 最大奖励: {np.max(total_rewards):.2f}")
        print(f"💾 模型文件: {self.model_path}")
        print("\\n💡 下一步: 运行 'python realtime_inference.py' 进行实时推理")


# ----------------------主训练函数----------------------
def main():
    """主训练函数"""
    print("=" * 60)
    print("🤖 Q-learning模型训练系统")
    print("=" * 60)

    # 训练Q-learning模型
    trainer = QLearningTrainer(MODEL_CONFIG)
    agent = trainer.train()

    print("\\n✅ 训练流程完成！")


if __name__ == "__main__":
    main()