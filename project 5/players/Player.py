class Player:
    """A Player

    The class that will represent each player in the game.
    A player can either be a Human or a DeepAgent.
    """
    def __init__(self, value, exploration_factor):
        self.val = value
        self.exploration_factor = exploration_factor

    def make_move(self, state, winner):
        i = int(input('Make a move: '))
        s = state[:i-1]+self.val+state[i:]
        return s
