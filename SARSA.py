import Board
import matplotlib.pyplot as plt

infinity = 999
average_bounce_list = list()

class SARSA():
    def __init__(self, episodes = 200000, n_threshold = 10, lr_constant = 30000, discount_factor = 0.98):
        self.board = Board.Board()
        self.q_table = dict()
        self.n_table = dict()
        self.episodes = episodes
        self.n_threshold = n_threshold
        self.lr_constant = lr_constant
        self.discount_factor = discount_factor

    def get_qvalue(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        return self.q_table[state][action]

    def set_qvalue(self, state, action, value):
        self.q_table[state][action] = value

    def get_nvalue(self, state, action):
        if state not in self.n_table:
            self.n_table[state] = [0, 0, 0]
        return self.n_table[state][action]

    def update_ntable(self, state, action):
        if state not in self.n_table:
            self.n_table[state] = [0, 0, 0]
        self.n_table[state][action] += 1

    def exploration_function(self, state, action):
        if self.get_nvalue(state, action) <= self.n_threshold:
            return infinity
        else:
            return self.get_qvalue(state, action)

    def get_action(self, state):
        max_qvalue = -infinity
        max_action = -1
        for action in range(3):
            current_value = self.exploration_function(state, action)
            if current_value > max_qvalue:
                max_qvalue = current_value
                max_action = action
        return max_action

    def get_maxQ_action(self, state):
        max_qvalue = -infinity
        max_action = -1
        for action in range(3):
            current_value = self.get_qvalue(state, action)
            if current_value > max_qvalue:
                max_qvalue = current_value
                max_action = action
        return max_action

    def run(self):
        bounces = 0
        start = True
        while True:
            current_state = self.board.convert()
            if start:
                current_action = self.get_maxQ_action(current_state)
                start = False
            else:
                current_action = next_action
            current_qvalue = self.get_qvalue(current_state, current_action)
            self.update_ntable(current_state, current_action)
            reward = self.board.update_board(current_action)

            next_state = self.board.convert()
            next_action = self.get_action(next_state)
            next_qvalue = self.get_qvalue(next_state, next_action)

            lr = self.lr_constant / (self.lr_constant + self.get_nvalue(current_state, current_action))
            new_qvalue = current_qvalue + lr * (reward + self.discount_factor * next_qvalue - current_qvalue)
            self.set_qvalue(current_state, current_action, new_qvalue)

            if reward == -1:
                return bounces

            bounces += reward

    def train(self):
        summation = 0
        for episode in range(self.episodes):
            self.board = Board.Board()
            bounces = self.run()
            summation += bounces
            average_bounce = summation / (episode + 1)
            average_bounce_list.append(average_bounce)
            if (episode + 1) % 1000 == 0:
                print ("For episode", episode + 1, "ball bounces", average_bounce, "times in average")
                print ("The difference is", average_bounce - average_bounce_list[-1000])

    def test(self):
        bounces = 0
        while True:
            current_state = self.board.convert()
            current_action = self.get_maxQ_action(current_state)
            reward = self.board.update_board(current_action)

            if reward == -1:
                return bounces

            bounces += reward

if __name__ == "__main__":
    sarsa = SARSA()
    print ("=======Training=======")
    sarsa.train()
    print ("=======Testing=======")
    average_bounce = 0
    for i in range(200):
        sarsa.board = Board.Board()
        average_bounce += sarsa.test() / 200
    print ("The average bounce for 200 test games is", average_bounce)
    print (sarsa.n_threshold, sarsa.lr_constant, sarsa.discount_factor)

    fig = plt.figure()
    plt.plot(list(range(1, sarsa.episodes + 1, 1)), average_bounce_list)
    plt.title("Mean Episode Rewards VS Episodes for SARSA")
    fig.savefig("Mean Episode Rewards VS Episodes for SARSA.png")

    plt.close(fig)
    print ("Saving Q-table now...")
    with open("SARSA-table.txt", "w") as f:
        for item in sarsa.q_table:
            f.write(str(item) + ":" + str(sarsa.q_table[item]) + "\n")
