import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import pygame

DARK_GRAY = (169, 169, 169)
BLACK = (0, 0, 0) 
PERSIMMON = (255, 77, 64)
YELLOW_GREEN = (154, 205, 50)
OLIVE_GREEN = (85, 107, 47)

# 導入物件
from snake_object import Snake 
from food_object import Food 

# Register this module as a gym environment.
register(
    id='SnakeGame-v0',                   # 貪食蛇環境的 ID
    entry_point='snake_env:SnakeEnv',    # module_name:class_name
)

# 實作我們的 Gym 環境，必須繼承自 gym.Env
class SnakeEnv(gym.Env):
    ACTION_TO_DIRECTION = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    
    metadata = {"render_modes": ["human"], 'render_fps': 10} 

    # 1. 初始化 (Initialization) - 體現封裝
    def __init__(self, size=10, render_mode=None):
        super().__init__()
        
        self.grid_size = size
        self.render_mode = render_mode
        
        # 實例化物件 (封裝)
        self.snake = Snake(size) 
        self.food = Food(size) # 實例化 Food 類別
        
        # 動作空間：上(0)、下(1)、左(2)、右(3)
        self.action_space = spaces.Discrete(4) 
        
        # 觀察空間：原本是座標，改成回傳 11 個布林值 (相對視角)
        # [危險直/右/左, 方向左/右/上/下, 食物左/右/上/下]
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(11,),
            dtype=np.int8
        )
        self.current_step = 0
        self.max_steps = size * size * 2 

        # 初始化 pygame 相關屬性
        self.window = None
        self.clock = None

    # 2. 重置環境 (Reset) - 體現抽象
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 

        # 重新實例化來重置狀態 
        self.snake = Snake(self.grid_size) 
        # 重生食物，確保不在蛇身上
        self.food = Food(self.grid_size) 
        
        
        try:
            body_pos = self.snake.get_body_positions()
        except AttributeError:
            body_pos = self.snake.body
            
        self.food.respawn(body_pos)
        
        # 構建初始觀察狀態 
        obs = self._get_observation()
        info = {}
        self.current_step = 0

        if(self.render_mode=='human'):
            self.render()

        return obs, info

    # 3. 執行動作 (Step) - 體現封裝與規則判斷
    def step(self, action):
        self.current_step += 1
        
        # 1. 動作執行
        direction_str = self.ACTION_TO_DIRECTION[action]
        self.snake.set_direction(direction_str) 
        self.snake.move()                     

        # 2. 判斷獎勵與終止
        head_pos = self.snake.get_head_positions() # 注意是複數
        food_pos = self.food.get_position()
        
        reward = 0
        terminated = False
        truncated = False 

        # 吃到食物
        if head_pos == food_pos:
            reward = 10   
            self.snake.grow() 
            
            # 取得身體位置給 Food 重生用
            try:
                body_pos = self.snake.get_body_positions()
            except AttributeError:
                body_pos = self.snake.body
                
            self.food.respawn(body_pos) 
        
        # 撞牆或吃到自己
        elif self.snake.is_touching_wall() or self.snake.is_touching_self():
            reward = -10 
            terminated = True
            
        # 只是移動
        else:
            reward = -0.1 

        # 步數限制 (Truncated 判斷)
        if self.current_step >= self.max_steps:
            terminated = True # 超過步數算結束，不然會跑太久
            truncated = True
        
        # 3. 構建新的觀察狀態 (相對視角)
        obs = self._get_observation()
        info = {}

        if(self.render_mode=='human'):
            self.render()

        return obs, reward, terminated, truncated, info

    # 這是為了 Q-Learning 能夠訓練加上的
    # 算出 11 個狀態回傳給 Agent
    def _get_observation(self):
        head = self.snake.get_head_positions() # 蛇頭
        
        # 算出蛇頭周圍四個點的座標
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        # 判斷蛇現在的方向
        dir_l = self.snake.direction == (-1, 0)
        dir_r = self.snake.direction == (1, 0)
        dir_u = self.snake.direction == (0, -1)
        dir_d = self.snake.direction == (0, 1)

        state = [
            # Danger Straight: 前方有危險嗎 (撞牆或撞身體)
            (dir_r and self._is_danger(point_r)) or 
            (dir_l and self._is_danger(point_l)) or 
            (dir_u and self._is_danger(point_u)) or 
            (dir_d and self._is_danger(point_d)),

            # Danger Right: 右手邊有危險嗎
            (dir_u and self._is_danger(point_r)) or 
            (dir_d and self._is_danger(point_l)) or 
            (dir_l and self._is_danger(point_u)) or 
            (dir_r and self._is_danger(point_d)),

            # Danger Left: 左手邊有危險嗎
            (dir_d and self._is_danger(point_r)) or 
            (dir_u and self._is_danger(point_l)) or 
            (dir_r and self._is_danger(point_u)) or 
            (dir_l and self._is_danger(point_d)),
            
            # Move Direction: 蛇現在的方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food Location: 食物在蛇的哪個相對方位
            self.food.position[0] < head[0],  # Food Left
            self.food.position[0] > head[0],  # Food Right
            self.food.position[1] < head[1],  # Food Up
            self.food.position[1] > head[1]   # Food Down
        ]
        
        return np.array(state, dtype=np.int8) # 轉成 numpy array 回傳

    # 輔助函式：檢查是不是撞到牆壁或身體
    def _is_danger(self, point):
        x, y = point
        # 檢查有沒有出界
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        
        # 檢查有沒有撞到身體 (排除蛇頭)
        try:
            body = list(self.snake.get_body_positions())
        except AttributeError:
            body = list(self.snake.body)

        if point in body[1:]:
            return True
        return False

    # 4. 畫面繪製 (Render)
    def render(self):
        if self.render_mode == "human":
            # 初始化 pygame window
            if self.window is None:
                pygame.init()
                pygame.display.init()
                PIXEL_SIZE = 40
                self.window_size = self.grid_size * PIXEL_SIZE
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock = pygame.time.Clock()
                pygame.display.set_caption("Greedy Snake")
            self.window.fill(BLACK)
            # 單一網格單元的尺寸
            cell_size = self.window_size // self.grid_size
            
            # 畫食物
            food_pos = self.food.get_position()
            food_x, food_y = food_pos
            # 繪製食物方塊 (PERSIMMON 柿子橙)
            pygame.draw.rect(
                self.window,
                PERSIMMON,
                (food_x * cell_size, food_y * cell_size, cell_size, cell_size)
            )

            # 畫蛇蛇
            try:
                snake_body = self.snake.get_body_positions()
            except AttributeError:
                snake_body = self.snake.body

            for i, (x, y) in enumerate(snake_body):
                # 蛇頭用綠色顯示
                color = OLIVE_GREEN if i == 0 else YELLOW_GREEN 
                
                # 蛇的身體
                pygame.draw.rect(
                    self.window,
                    color,
                    (x * cell_size, y * cell_size, cell_size, cell_size)
                )

                # 可選：繪製網格線
                pygame.draw.rect(
                    self.window,
                    DARK_GRAY,
                    (x * cell_size, y * cell_size, cell_size, cell_size),
                    1 # 邊框厚度為 1 像素
                )

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            

    # 5. 釋放資源 (Close)
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

if __name__=="__main__":
    env = gym.make('SnakeGame-v0', render_mode='human') 

    # 以下是測試用
    obs, info = env.reset()
    print("Initial State (11 dim):", obs) 

    terminated = False
    truncated = False
    total_reward = 0
    
    # 跑個 500 步看看
    for _ in range(500):
        if terminated or truncated:
            break

        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Pygame 事件處理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
    print(f"遊戲結束!總步數: {env.unwrapped.current_step}, 總獎勵: {total_reward:.2f}")

    env.close()
