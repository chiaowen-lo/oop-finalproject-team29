from random import randint

class Food:
    def __init__(self, board_size):
        self.board_size = board_size

        # 在隨幾位置上先生成第一個 Food
        rand_x = randint(0, self.board_size - 1)
        rand_y = randint(0, self.board_size - 1)
        self.position = (rand_x, rand_y)

    # 回傳食物當前的座標
    def get_position(self):
        return self.position

    def _generate_random_position(self, unavailable_position):
        while True:
            rand_x = randint(0, self.board_size - 1)
            rand_y = randint(0, self.board_size - 1)
            if (rand_x, rand_y) not in unavailable_position:
                break
        self.position = (rand_x, rand_y)
    
    def respawn(self, snake_body_positions):
        self._generate_random_position(snake_body_positions)
