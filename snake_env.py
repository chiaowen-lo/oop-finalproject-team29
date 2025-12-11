import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np

# 導入物件
from .snake_object import Snake 
from .food_object import Food 

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
        
        # 實例化組員 C 的物件 (封裝)
        self.snake = Snake(size) 
        self.food = Food(size) # 實例化 Food 類別
        
        # 動作空間：上(0)、下(1)、左(2)、右(3)
        self.action_space = spaces.Discrete(4) 
        
        # 觀察空間：[蛇頭X, 蛇頭Y, 食物X, 食物Y]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_size-1, self.grid_size-1, self.grid_size-1, self.grid_size-1]),
            shape=(4,),
            dtype=np.int32
        )
        self.current_step = 0
        self.max_steps = size * size * 2 

    # 2. 重置環境 (Reset) - 體現抽象
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 

        # 重新實例化來重置狀態 (最簡單的重置方法)
        self.snake = Snake(self.grid_size) 
        # 重生食物，確保不在蛇身上
        self.food = Food(self.grid_size) 
        self.food.respawn(self.snake.body)
        
        # 構建初始觀察狀態
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food.get_position() 
        
        obs = np.array([head_x, head_y, food_x, food_y], dtype=np.int32)
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
        head_pos = self.snake.get_head_position()
        food_pos = self.food.get_position()
        
        reward = 0
        terminated = False
        truncated = False 

        # 吃到食物
        if head_pos == food_pos:
            reward = 10   
            self.snake.grow() 
            self.food.respawn(self.snake.body) # 呼叫 Food 的重生邏輯
        
        # 撞牆或吃到自己
        elif self.snake.is_touching_wall() or self.snake.is_touching_self():
            reward = -10 
            terminated = True
            
        # 只是移動
        else:
            reward = -0.1 

        # 步數限制 (Truncated 判斷)
        if self.current_step >= self.max_steps:
            terminated = True
            truncated = True
        
        # 3. 構建新的觀察狀態
        new_food_x, new_food_y = self.food.get_position()
        new_head_x, new_head_y = head_pos
        
        obs = np.array([new_head_x, new_head_y, new_food_x, new_food_y], dtype=np.int32)
        info = {}

        if(self.render_mode=='human'):
            self.render()

        return obs, reward, terminated, truncated, info

    # 4. 畫面繪製 (Render)
    def render(self):
        # 呼叫組員 C 的繪製邏輯
        # self.snake.render() 
        # self.food.render()
        pass

    # 5. 釋放資源 (Close)
    def close(self):
        pass

if __name__=="__main__":
    env = gym.make('SnakeGame-v0', render_mode='human') 
    # ... (省略測試代碼)
    env.close()