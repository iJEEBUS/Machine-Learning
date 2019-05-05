from engine.ttt import TicTacToe
from players import DeepAgent as model

player_one = model.DeepAgent('X', 0.9)
player_two = model.DeepAgent('O', 0.9)
game = TicTacToe(player_one, player_two)
game.play_game()
