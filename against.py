import math
import numpy as np

infinity = 999

def readData(inputFilePath):
    file = open(inputFilePath, "r")
    lines = file.readlines()
    boardData = list()
    choiceData = list()
    for line in lines:
        cache  = line.strip('\n')
        cache = line.split(' ')
        board_temp = list()
        for index, item in enumerate(cache):
            if index != 5:
                board_temp.append(float(item))
            else:
                choiceData.append(float(item))
        boardData.append(board_temp)
    return boardData, choiceData

class againstBoard():
    def __init__(self, ball_x = 0.5, ball_y = 0.5, velocity_x = 0.03, velocity_y = 0.01, paddle_y_A = 0.5, paddle_y_B = 0.5):
        '''
        paddle_A is at x = 0, paddle_B is at x = 1
        '''
        self.paddle_height = 0.2
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.paddle_y_A = paddle_y_A - self.paddle_height / 2
        self.paddle_y_B = paddle_y_B - self.paddle_height / 2
        self.action_list = [0, +0.04, -0.04]

    def update_board(self, actionA, actionB):
        # update the paddle coordinate
        self.paddle_y_A += self.action_list[actionA]
        if self.paddle_y_A >= 1 - self.paddle_height:
            self.paddle_y_A = 1 - self.paddle_height
        elif self.paddle_y_A <= 0:
            self.paddle_y_A = 0

        self.paddle_y_B += self.action_list[actionB]
        if self.paddle_y_B >= 1 - self.paddle_height:
            self.paddle_y_B = 1 - self.paddle_height
        elif self.paddle_y_B <= 0:
            self.paddle_y_B = 0

        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y
        # off top
        if self.ball_y < 0:
            self.ball_y = -self.ball_y
            self.velocity_y = -self.velocity_y
        # off bottom
        if self.ball_y > 1:
            self.ball_y = 2 - self.ball_y
            self.velocity_y = -self.velocity_y

        # the ball is still within the boundary
        if self.ball_x < 1 and self.ball_x > 0:
            return (0,0)
        # the ball misses the paddle_A
        if (self.ball_y < self.paddle_y_A or self.ball_y > self.paddle_y_A + self.paddle_height) and (self.ball_x < 0):
            return (-1,0)
        # the ball misses the paddle_B
        if (self.ball_y < self.paddle_y_B or self.ball_y > self.paddle_y_B + self.paddle_height) and (self.ball_x > 1):
            return (0,-1)
        # the ball hits the paddle_A
        if (self.ball_x < 0) and (self.ball_y >= self.paddle_y_A and self.ball_y <= self.paddle_y_A + self.paddle_height):
            self.ball_x = -self.ball_x
            u = np.random.uniform(-0.015, 0.015)
            v = np.random.uniform(-0.03, 0.03)
            # making sure |velocity_x| > 0.03 and |velocity_x| < 1
            while abs(-self.velocity_x + u) <= 0.03 or abs(-self.velocity_x + u) >= 1:
                u = np.random.uniform(-0.015, 0.015)
            # making sure |velocity_y| < 1
            while abs(self.velocity_y + v) >= 1:
                u = np.random.uniform(-0.015, 0.015)
            self.velocity_x = -self.velocity_x + u
            self.velocity_y = self.velocity_y + v
            return (1,0)
        # the ball hits the paddle_B
        if (self.ball_x > 1) and (self.ball_y >= self.paddle_y_B and self.ball_y <= self.paddle_y_B + self.paddle_height):
            self.ball_x = 2 - self.ball_x
            u = np.random.uniform(-0.015, 0.015)
            v = np.random.uniform(-0.03, 0.03)
            # making sure |velocity_x| > 0.03 and |velocity_x| < 1
            while abs(-self.velocity_x + u) <= 0.03 or abs(-self.velocity_x + u) >= 1:
                u = np.random.uniform(-0.015, 0.015)
            # making sure |velocity_y| < 1
            while abs(self.velocity_y + v) >= 1:
                u = np.random.uniform(-0.015, 0.015)
            self.velocity_x = -self.velocity_x + u
            self.velocity_y = self.velocity_y + v
            return (0,1)
    '''
    This function is used for paddle_A to convert to discrete state
    '''
    def convert(self):
        # The special state where the ball is out of boundary
        if (self.ball_x < 0):
            return (12,12,12,12,12)

        new_ball_x = int(math.floor(12 * self.ball_x))
        if new_ball_x == 12:
            new_ball_x = 11
        new_ball_y = int(math.floor(12 * self.ball_y))
        if new_ball_y == 12:
            new_ball_y = 11
        new_velocity_x = int(abs(self.velocity_x) / self.velocity_x)
        new_velocity_y = int(abs(self.velocity_y) / self.velocity_y)
        if abs(self.velocity_y) < 0.015:
            new_velocity_y = 0
        new_paddle_y_A = int(math.floor(12 * self.paddle_y_A / (1 - self.paddle_height)))
        if (self.paddle_y_A == 1 - self.paddle_height):
            new_paddle_y_A = 11

        return (new_ball_x, new_ball_y, new_velocity_x, new_velocity_y, new_paddle_y_A)

class againstGame():
    def __init__(self, averg_param, std_param):
        self.board = againstBoard()
        self.q_table = dict()
        self.weight, self.bias = list(), list()
        self.averg_param = averg_param
        self.std_param = std_param

    def get_qvalue(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        return self.q_table[state][action]

    def get_maxQ_action(self, state):
        max_qvalue = -infinity
        max_action = -1
        for action in range(3):
            current_value = self.get_qvalue(state, action)
            if current_value > max_qvalue:
                max_qvalue = current_value
                max_action = action
        return max_action

    def affine_forward(self, A, W, b):
        acache = (A, W, b)
        new_A = np.asarray(A)
        new_W = np.asarray(W)
        new_B = np.tile(np.asarray(b), (len(A), 1))
        Z = np.add(np.dot(new_A, new_W), new_B)
        return (Z.tolist(), acache)

    def relu_forward(self, Z):
        new_Z = np.asarray(Z)
        output = np.maximum(new_Z, 0)
        return (output.tolist(), Z)

    def four_network(self, x):
        z1, acache1 = self.affine_forward(x, self.weight[0], self.bias[0])
        a1, rcache1 = self.relu_forward(z1)
        z2, acache2 = self.affine_forward(a1, self.weight[1], self.bias[1])
        a2, rcache2 = self.relu_forward(z2)
        z3, acache3 = self.affine_forward(a2, self.weight[2], self.bias[2])
        a3, rcache3 = self.relu_forward(z3)
        f, acache4 = self.affine_forward(a3, self.weight[3], self.bias[3])

        for item in f:
            classification = item.index(max(item))
        return classification

    def compete(self):
        bounceA, bounceB = 0,0
        while True:
            current_state_A = self.board.convert()
            current_action_A = self.get_maxQ_action(current_state_A)

            ball_x = (self.board.ball_x - self.averg_param[0]) / self.std_param[0]
            ball_y = (self.board.ball_y - self.averg_param[1]) / self.std_param[1]
            velocity_x = (self.board.velocity_x - self.averg_param[2]) / self.std_param[2]
            velocity_y = (self.board.velocity_y - self.averg_param[3]) / self.std_param[3]
            paddle_y_B = (self.board.paddle_y_B - self.averg_param[4]) / self.std_param[4]
            curr_action = self.four_network([ball_x, ball_y, velocity_x, velocity_y, paddle_y_B])
            current_action_B = (curr_action + 2) % 3

            rewardA, rewardB = self.board.update_board(current_action_A, current_action_B)
            if rewardA == -1:
                # print ("B wins")
                # print ("The bounce times for A and B are", bounceA, "and", bounceB)
                return 2, bounceA, bounceB
            elif rewardB == -1:
                # print ("A wins")
                # print ("The bounce times for A and B are", bounceA, "and", bounceB)
                return 1, bounceA, bounceB

            bounceA += rewardA
            bounceB += rewardB

if __name__ == "__main__":

    q_table = dict()
    with open("NEW Q-table.txt", "r") as f:
        for line in f:
            line = line.rstrip('\n')
            entry = line.split(":")
            q_table[eval(entry[0])] = eval(entry[1])

    expected_weight = list()
    expected_bias = list()
    with open("orig-dnn-weight.txt", "r") as f:
        for line in f:
            line = line.rstrip("\n")
            expected_weight.append(list(eval(line)))

    with open("orig-dnn-bias.txt", "r") as f:
        for line in f:
            line = line.rstrip("\n")
            expected_bias.append(list(eval(line)))

    batchData, batchTarget = readData("expert_policy.txt")
    average = np.mean(np.asarray(batchData), axis = 0)
    std = np.std(np.asarray(batchData), axis = 0)

    game = againstGame(average, std)
    game.q_table = q_table
    game.weight = expected_weight
    game.bias = expected_bias

    A_wins, B_wins = 0,0
    A, B = 0,0
    episodes = 1000

    for i in range(episodes):
        game.board = againstBoard()
        print ("============Round", i + 1, "============")
        result, bounceA, bounceB = game.compete()
        if result == 1:
            A_wins += 1
        else:
            B_wins += 1
        A += bounceA / episodes
        B += bounceB / episodes

    print ("In 1000 games, A wins", A_wins, "times, B wins", B_wins, "times.")
    print ("Average bounce for A is", A)
    print ("Average bounce for B is", B)

