from collections import deque

class Snake:
    # UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3
    # left top = (0, 0)
    DIRECTIONS = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0)
    }

    def __init__(self, board_size, start_pos = (3, 0), start_dir='RIGHT', initial_length=3):
        # board size (0 ~ N-1)
        # 正方形地圖：輸入整數Ｎ，地圖大小 N * N   <- 這個
        # 長方形地圖：輸入(W, H)，地圖大小 W * H
        self.board_size = board_size
        self.direction = self.DIRECTIONS[start_dir] # 蛇蛇的移動方向
        self._growing = False # 蛇蛇是否要變長

        # 蛇蛇的身體跟初始位置
        self.body = deque()
        current_x, current_y = start_pos
        dx, dy = self.DIRECTIONS[start_dir]
        for i in range(initial_length):
            self.body.append((current_x, current_y))
            current_x -= dx
            current_y -= dy

    # 回傳蛇蛇的頭的位置
    def get_head_positions(self):
        return self.body[0]
    
    # 回傳蛇蛇的長度（獎勵機制的分數用）
    def get_length(self):
        return len(self.body)
    
    # 回傳蛇身位置 (檢查 Food有沒有跟 body重疊用)
    def get_body_positions(self):
        return list(self.body)
    
    # 蛇吃到食物時呼叫
    def grow(self):
        self._growing = True

    # 檢查蛇的頭是否撞到邊界
    def is_touching_wall(self):
        head_x, head_y = self.get_head_positions()
        N = self.board_size # 地圖邊界大小
        if head_x < 0 or head_x >= N or head_y < 0 or head_y >= N:
            return True
        return False
    
    # 檢查蛇蛇是不是碰到自己
    def is_touching_self(self):
        head = self.get_head_positions()
        if head in list(self.body)[1:]:
            return True
        return False
    
    # 蛇蛇移動
    # 移動方式：頭朝著direction往前一格，刪除尾巴
    # 如果正在 growing，就不刪除尾巴
    def move(self):
        # 計算新的蛇頭座標
        dx, dy = self.direction
        head_x, head_y = self.get_head_positions()
        new_head_x, new_head_y = head_x + dx, head_y + dy

        # 把新的蛇頭加進身體
        self.body.appendleft((new_head_x, new_head_y))

        # 判斷這次 move 要不要刪除尾巴
        if self._growing:
            self._growing = False 
        else:
            self.body.pop()

    # 蛇蛇 Agent更新移動方向時呼叫(確保蛇不會180度反轉)
    def set_direction(self, new_direction):
        new_dx, new_dy = self.DIRECTIONS[new_direction]
        current_dx, current_dy = self.direction
        if new_dx + current_dx == 0 and new_dy + current_dy == 0: # 原方向跟新方向差了180度（反向）
            return
        self.direction = (new_dx, new_dy)