from ChessEnv import ChessEnv
from QLearning import QLearning

if __name__ == "__main__":
    env = ChessEnv()
    actions = env.get_legal_actions()
    q_learning = QLearning(actions)
    data = env.generate_dataset(q_learning, episodes=1000)
    env.train_supervised(data)