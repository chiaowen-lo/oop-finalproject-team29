import gymnasium as gym
import numpy as np
import pickle
import time
import os
import pygame 
from collections import deque
import matplotlib.pyplot as plt 
import snake_env 

class QLearningAgent:
    def __init__(self, action_space_n, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9997, epsilon_min=0.01):
        self.action_space_n = action_space_n
        self.q_table = {} 
        self.lr = alpha          
        self.gamma = gamma       
        self.epsilon = epsilon   
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min

    def get_state_key(self, state):
        return tuple(state)

    def get_action(self, state, env=None, force_greedy=False, use_flood_fill=False):
        """ 
        force_greedy=True: 測試模式 (完全不隨機)
        use_flood_fill=True: 開啟洪水填充演算法 (僅在 Test B / Demo 使用)
        """
        state_key = self.get_state_key(state)
        
        # 取得 Q 值 (如果沒看過就初始化為 0)
        if state_key not in self.q_table:
            q_values = np.zeros(self.action_space_n)
        else:
            q_values = self.q_table[state_key].copy()

        # --- 1. 訓練模式 / 隨機探索 ---
        if not force_greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_n)

        # --- 2. 增強測試模式 (Flood Fill) ---
        if use_flood_fill and env is not None:
            # 將動作依照 Q 值由大到小排序
            sorted_actions = np.argsort(q_values)[::-1]
            
            head_x, head_y = env.unwrapped.snake.get_head_positions()
            body_set = set(env.unwrapped.snake.get_body_positions())
            grid_size = env.unwrapped.grid_size
            
            # 動作對應：0:上, 1:下, 2:左, 3:右
            moves = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

            for action in sorted_actions:
                dx, dy = moves[action]
                nx, ny = head_x + dx, head_y + dy
                
                # (A) 基礎檢查 (撞牆或撞身體)
                if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                    continue
                if (nx, ny) in body_set:
                    continue
                
                # (B) 洪水填充檢查 (空間是否足夠)
                if self._check_flood_fill(env, nx, ny, body_set):
                    return action 
            
            # 如果都不安全，回傳 Q 值最高的 (聽天由命)
            return np.argmax(q_values)

        # --- 3. 一般測試模式 ---
        return np.argmax(q_values)

    def _check_flood_fill(self, env, start_x, start_y, body_set):
        """ 使用 BFS 演算法檢查從 (start_x, start_y) 開始，是否有足夠的空間 """
        grid_size = env.unwrapped.grid_size
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        temp_body = body_set.copy()
        temp_body.add((start_x, start_y))
        
        queue = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        count = 0
        limit = len(body_set) * 2 # 搜尋範圍限制
        
        while queue:
            cx, cy = queue.popleft()
            count += 1
            if count >= limit:
                return True # 空間足夠
            
            for dx, dy in moves:
                tx, ty = cx + dx, cy + dy
                if (0 <= tx < grid_size and 
                    0 <= ty < grid_size and 
                    (tx, ty) not in temp_body and 
                    (tx, ty) not in visited):
                    visited.add((tx, ty))
                    queue.append((tx, ty))
        
        return False # 空間不足 (死路)

    def update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space_n)

        predict = self.q_table[state_key][action]
        if done:
            target = reward 
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])
            
        self.q_table[state_key][action] += self.lr * (target - predict)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename="q_table_20x20.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename="q_table_20x20.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded: {filename}")
            return True
        return False

# --- 輔助函式：執行測試並收集數據 ---
def run_test_session(agent, env, episodes, use_flood_fill, target_reward):
    success_count = 0
    total_rewards = []
    
    # 指標列表
    efficiency_list = [] # 效率 = 步數 / 食物
    length_list = []     # 最終長度
    
    config_msg = "ON" if use_flood_fill else "OFF"
    print(f"   Flood Fill: {config_msg}")
    
    for i in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        # 追蹤單局數據
        episode_steps = 0
        episode_food = 0
        
        while not done:
            action = agent.get_action(obs, env=env, force_greedy=True, use_flood_fill=use_flood_fill)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            episode_steps += 1
            if reward > 5: # 假設吃到食物獎勵 > 5
                episode_food += 1
            
        total_rewards.append(episode_reward)
        if episode_reward > target_reward:
            success_count += 1
            
        # 計算長度 (初始 3 + 吃到數量)
        final_len = 3 + episode_food
        length_list.append(final_len)
        
        # 計算效率 (Efficiency = Total_Steps / Food_Eaten)
        if episode_food > 0:
            eff = episode_steps / episode_food
            efficiency_list.append(eff)
        
        if (i + 1) % 200 == 0:
            print(f"      Progress: {i + 1}/{episodes}...")
            
    avg_reward = np.mean(total_rewards)
    success_rate = (success_count / episodes) * 100
    
    # 計算平均值
    avg_length = np.mean(length_list) if length_list else 3.0
    avg_efficiency = np.mean(efficiency_list) if efficiency_list else 0.0
    
    return success_rate, avg_reward, avg_length, avg_efficiency

# --- 繪製訓練曲線函式 (Moving Average) ---
def plot_training_curve(rewards, filename="training_curve.png", window=100):
    plt.figure(figsize=(10, 6))
    
    # 計算移動平均 (讓曲線變平滑，不然會太亂)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f'Moving Average (Window={window})', color='blue')
    else:
        plt.plot(rewards, label='Total Reward', color='alpha_blue')

    plt.title('Training Progress: Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Training curve saved as '{filename}'")
    plt.close() # 關閉圖表釋放記憶體

# --- 主程式 ---
if __name__ == "__main__":
    
    MAP_SIZE = 20
    TRAIN_EPISODES = 15000 
    TEST_EPISODES = 1000
    TARGET_REWARD = 200 
    MODEL_FILE = "q_table_20x20.pkl"
    LOG_FILE = "training_log.txt"

    # ==========================
    # 1. 訓練階段 (Training Phase)
    # ==========================
    env = gym.make('SnakeGame-v0', render_mode=None, size=MAP_SIZE)
    agent = QLearningAgent(action_space_n=env.action_space.n, epsilon_decay=0.9997)

    # 用來畫圖的陣列
    training_rewards = []

    # 開啟 Log 檔案準備寫入
    with open(LOG_FILE, "w") as f:
        f.write("Episode,Reward,Epsilon\n") # 寫入標題

        print(f"[Step 1] Training... {TRAIN_EPISODES} Episodes")
        
        for episode in range(TRAIN_EPISODES):
            state, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, term, trunc, info = env.step(action)
                done = term or trunc
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward # 累加分數
            
            agent.decay_epsilon()
            
            # 紀錄數據
            training_rewards.append(total_reward)
            
            # 每回合寫入 Log (或是每 10 回合寫一次也可以，這裡每回合都寫)
            f.write(f"{episode+1},{total_reward:.2f},{agent.epsilon:.5f}\n")

            if (episode + 1) % 1000 == 0:
                print(f"   Ep {episode+1}, Epsilon: {agent.epsilon:.4f}, Last Reward: {total_reward:.2f}")

    print(f"Training Completed! Log saved to '{LOG_FILE}'")
    agent.save_model(MODEL_FILE)
    
    # 🔥 訓練結束後，馬上畫訓練曲線圖
    plot_training_curve(training_rewards, filename="training_curve.png")
    
    env.close()

    # ==========================
    # 2. 比較測試 (Comparison Testing)
    # ==========================
    print(f"\n[Step 2] Comparison Testing ({TEST_EPISODES} Episodes)")
    test_env = gym.make('SnakeGame-v0', render_mode=None, size=MAP_SIZE)
    agent.load_model(MODEL_FILE)

    # Test A: 基礎版 (Basic)
    print(f"\nTest A: Basic Q-Learning")
    rate_a, reward_a, len_a, eff_a = run_test_session(agent, test_env, TEST_EPISODES, False, TARGET_REWARD)
    print(f"   [Result] Success: {rate_a:.1f}%, Length: {len_a:.1f}, Efficiency: {eff_a:.1f} steps/food")

    # Test B: 增強版 (Flood Fill)
    print(f"\nTest B: Flood Fill Enhanced")
    rate_b, reward_b, len_b, eff_b = run_test_session(agent, test_env, TEST_EPISODES, True, TARGET_REWARD)
    print(f"   [Result] Success: {rate_b:.1f}%, Length: {len_b:.1f}, Efficiency: {eff_b:.1f} steps/food")
    
    test_env.close()

    # ==========================
    # 3. 視覺化比較圖表 (Visualization)
    # ==========================
    print("\nGenerating Comparison Charts...")
    
    labels = ['Basic', 'Flood Fill']
    x = np.arange(len(labels))
    width = 0.5
    
    # 1 列 3 行的比較圖
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Chart 1: 成功率
    success_rates = [rate_a, rate_b]
    rects1 = ax1.bar(x, success_rates, width, color='#88c999')
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate (Higher is Better)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 105)
    ax1.bar_label(rects1, padding=3, fmt='%.1f%%')

    # Chart 2: 平均長度
    avg_lengths = [len_a, len_b]
    rects2 = ax2.bar(x, avg_lengths, width, color='#66b3ff')
    ax2.set_ylabel('Avg Snake Length', fontsize=12)
    ax2.set_title('Average Length (Higher is Better)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.bar_label(rects2, padding=3, fmt='%.1f')
    
    # Chart 3: 效率
    efficiencies = [eff_a, eff_b]
    rects3 = ax3.bar(x, efficiencies, width, color='#ff9999')
    ax3.set_ylabel('Steps per Food', fontsize=12)
    ax3.set_title('Efficiency (Lower is Better)', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.bar_label(rects3, padding=3, fmt='%.1f')

    plt.tight_layout()
    plt.savefig('comparison_chart.png')
    print("Chart saved as 'comparison_chart.png'")

    # ==========================
    # 4. 最終 Demo (Final Demo)
    # ==========================
    input("\nPress [Enter] for Final Demo...")
    demo_env = gym.make('SnakeGame-v0', render_mode='human', size=MAP_SIZE)
    obs, info = demo_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        
        # Demo 使用 Flood Fill
        action = agent.get_action(obs, env=demo_env, force_greedy=True, use_flood_fill=True)
        obs, reward, term, trunc, info = demo_env.step(action)
        done = term or trunc
        total_reward += reward
        
        if done:
            print("-" * 30)
            print(f"Demo Finished! Reward: {total_reward:.2f}")
            snake = demo_env.unwrapped.snake
            if trunc: print("Cause: Timeout (Starvation)")
            elif snake.is_touching_wall(): print("Cause: Hit Wall")
            elif snake.is_touching_self(): print("Cause: Body Collision")
            print("-" * 30)
            time.sleep(2)
    demo_env.close()
