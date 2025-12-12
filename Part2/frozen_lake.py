import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

#檔案設定
FILENAME = 'frozen_lake8x8.pkl'
MAP_FILE = 'frozen_lake_map.txt'
RESULT_FILE = 'result.txt'
PLOT_FILE = 'frozen_lake8x8.png' # 新增圖表檔名

def save_map_to_file(random_map, seed):
    """生成的地圖存入MAP_FILE"""
    with open(MAP_FILE, 'w') as f:
        f.write(f"Seed used: {seed}\n")
        f.write("-" * 20 + "\n")
        for row in random_map:
            f.write(row + '\n')
    print(f"地圖檔案已儲存至: {MAP_FILE}")

def print_map_preview(random_map):
    """Terminal顯示map"""
    print("\n" + "="*20)
    print("MAP PREVIEW")
    
    # 將地圖列表轉為字串顯示
    for _, row in enumerate(random_map):
        print(f"{row}")
    print("="*20 + "\n")

def evaluate_agent(q, map_desc, episodes=1000):
    """ 測試函式 """

    #隨機地圖來建立環境, 地板會滑, 不顯示畫面
    env = gym.make('FrozenLake-v1', desc=map_desc, is_slippery=True, render_mode=None)
    
    rewards = 0
    rng = np.random.default_rng() #亂數產生器
    
    for _ in range(episodes):
        state = env.reset()[0] #Agent放起點
        terminated = False #尚未結束
        truncated = False  #步數尚未耗盡

        while not terminated and not truncated:
            # Tie-breaking

            # 找出目前這一格 (State) 做什麼動作會拿比較高分
            max_val = np.max(q[state, :])

            # 找出所有分數並列第一的動作索引
            actions = np.flatnonzero(q[state, :] == max_val)

            # 並列第一的動作中，隨機挑一個
            action = rng.choice(actions)
            
            #執行選出的動作，看環境回傳什麼結果
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state

        if reward == 1: #走到終點G
            rewards += 1
    env.close()

    # 回傳勝率(成功次數 / 總次數) * 100
    return (rewards / episodes) * 100

def train_sarsa_agent(map_desc):
    """ SARSA 訓練函式 """

    #Setup(建立環境、初始化Q表)
    env = gym.make('FrozenLake-v1', desc=map_desc, is_slippery=True, render_mode=None)
    q = np.zeros((env.observation_space.n, env.action_space.n))

    episodes = 15000            #固定num_episodes
    learning_rate_a = 0.1       #Alpha: 學習率
    discount_factor_g = 0.99    #Gamma: 折扣因子
    
    epsilon = 1.0                 #探索率: 一開始 100% 隨機亂走
    epsilon_decay_rate = 0.0001   #衰減率: 慢慢減少隨機，變成依賴經驗
    min_exploration_rate = 0.01   #最低探索率

    rng = np.random.default_rng()
    
    # 建立一個陣列來記錄每一場有沒有贏
    rewards_history = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # 回到起點
        #Epsilon-Greedy
        if rng.random() < epsilon:
            """
            ---Exploration---
            防止Agent永遠只走它知道的路，強迫嘗試未知的區域
            """
            action = env.action_space.sample()

        else:
            """
            ---Exploitation---
            Agent 要根據它目前的經驗 (Q-Table) 來選最好的動作
            """

            #找出目前這一格最高的分數是多少
            max_val = np.max(q[state, :])
            actions = np.flatnonzero(q[state, :] == max_val)

            #從這些最高分的動作中，隨機挑一個
            action = rng.choice(actions)
            
        terminated = False #尚未結束
        truncated = False  #步數尚未耗盡

        while(not terminated and not truncated):
            #1. 執行動作 (Take Action) -> 得到 R, S'
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            #2. 選擇下一個動作A', 用Epsilon-Greedy
            if rng.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                max_val = np.max(q[new_state, :])
                actions = np.flatnonzero(q[new_state, :] == max_val)
                next_action = rng.choice(actions)


            #3. 更新 Q 表
            """
            # 公式: Q(S,A) = Q(S,A) + alpha * [Target - Q(S,A)]
            # Target = Reward + gamma * Q(S', A')
            """
            target = reward + discount_factor_g * q[new_state, next_action]
            q[state, action] = q[state, action] + learning_rate_a * (target - q[state, action])
            

            #4. 推進時間
            state = new_state
            action = next_action # 準備進入下一輪

        #衰減探索率
        epsilon = max(min_exploration_rate, epsilon - epsilon_decay_rate)
        
        # --- 修改點 2: 記錄這一場的結果 ---
        if reward == 1:
            rewards_history[i] = 1
        
    env.close()
    
    # --- 修改點 3: 回傳 q 表和歷史紀錄 ---
    return q, rewards_history

if __name__ == '__main__':

    # 清理舊檔
    if os.path.exists(FILENAME): os.remove(FILENAME)
    if os.path.exists(RESULT_FILE): os.remove(RESULT_FILE)
    if os.path.exists(MAP_FILE): os.remove(MAP_FILE)

    print("--- Map Scavenging with SARSA ---")
    print("目標：隨機生成地圖，直到找到success rate>70%的地圖\n")
    
    found_good_map = False  
    attempt = 0             #目前測試幾個
    max_attempts = 15       #上限10個地圖

    while not found_good_map and attempt < max_attempts: #尚未找到地圖且還沒到達上限
        attempt += 1
        # 1. 隨機生成新地圖
        current_seed = int(np.random.randint(0, 1000))
        # p=0.8 提高安全地機率，讓好地圖更容易出現
        current_map = generate_random_map(size=8, p=0.80, seed=current_seed)
        
        print(f"嘗試第 {attempt} 張地圖 (Seed: {current_seed})...", end=" ")
        
        # 2. 訓練 (接收兩個回傳值)
        q_table, train_rewards = train_sarsa_agent(current_map)
        
        # 3. 測試
        score = evaluate_agent(q_table, current_map, episodes=1000)
        print(f"成績: {score:.2f}%")
        
        # 4. 判斷是否合格
        if score >= 70.0:
            found_good_map = True
            print(f"\n找到合適地圖了！")
            
            # 顯示與存檔
            print_map_preview(current_map)      # 顯示地圖
            save_map_to_file(current_map, current_seed) # 存地圖
            

            # 存 AI 模型 (.pkl)
            with open(FILENAME, "wb") as f:
                pickle.dump(q_table, f)
            

            #寫result.txt
            result_content = (
                f"Model: SARSA\n"
                f"Map Seed: {current_seed}\n"
                f"Map File: {MAP_FILE}\n"
                f"Success Rate: {score:.2f}%\n"
                f"Parameters: LR=0.1, Gamma=0.99"
            )
            with open(RESULT_FILE, "w") as f:
                f.write(result_content)
                
            # 繪製學習曲線 ---
            # 計算移動平均 (Window=500)
            window_size = 500
            moving_avg = np.convolve(train_rewards, np.ones(window_size)/window_size, mode='valid')
            
            # 建立圖表
            plt.figure(figsize=(10, 6))
            plt.plot(moving_avg, label='Moving Average (Window=500)', color='blue')
            plt.title(f"Learning Curve (Seed: {current_seed})")
            plt.xlabel("Episodes")
            plt.ylabel("Success Rate")
            plt.legend()
            plt.grid(True)
            plt.savefig(PLOT_FILE) # 存成 png
            plt.close()            # 關閉
            
            print(f"所有檔案已輸出完成！")
            
        

    if not found_good_map:
        print("\n沒有找到好地圖，請重新執行程式。")
