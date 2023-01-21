import chess
import chess.engine
from sklearn.linear_model import LogisticRegression


class ChessEnv:
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish-windows-2022-x86-64-avx2")
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.board.fen()

    def step(self, action):
        self.board.push_uci(action)
        state = self.board.fen()
        legal_actions = self.get_legal_actions()
        next_action = self.get_action(state, legal_actions)
        if self.board.is_checkmate():
            reward = 1
            done = True
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition() or self.board.is_variant_draw():
            reward = 0.5
            done = True
        else:
            reward = 0
            done = False
        return state, next_action, reward, done

    def get_legal_actions(self):
        legal_moves = [move.uci() for move in self.board.legal_moves]
        return legal_moves

    def get_action(self, state, legal_actions):
        self.engine.set_position(self.board)
        result = self.engine.play(chess.engine.Limit(time=0.1))
        return result.move.uci()

    def generate_dataset(self, q_learning, episodes=1000):
        data = []
        for episode in range(episodes):
            state = self.reset()
            legal_actions = self.get_legal_actions()
            done = False
            while not done:
                action = legal_actions[q_learning.get_action(state)]
                next_state, next_action, reward, done = self.step(action)
                q_learning.learn(state, action, next_state, next_action, reward)
                state = next_state
                data.append((state, action))
        return data

    def train_supervised(self, data):
        """
        Train a supervised model on the generated dataset
        """
        X, y = zip(*data)
        model = LogisticRegression()
        model.fit(X, y)
        return model
