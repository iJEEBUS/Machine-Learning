import time # for timing execution
import random # to pick who goes first
import keras.layers as kl # for NN
import keras.models as km # for NN
import numpy as np
from players.Player import Player    
class TicTacToe():
    """The Game

    The actual logic for the TTT game is located in this class.
    """
    def __init__(self, p1, p2):
        """Constructor
        
        Creates a new TTT board and displays it.
        """
        
        # Create the state (board)
        self.state = '123456789'
        self.winning_combos = ([6, 7, 8], [3, 4, 5], 
                               [0, 1, 2], [0, 3, 6], 
                               [1, 4, 7], [2, 5, 8], 
                               [0, 4, 8], [2, 4, 6],)
        self.game_over = False
        
        # Create the players
        self.player_one = p1 
        self.player_two = p2 
        self.winner = None

        # Track the winners
        self.x_wins = 0
        self.o_wins = 0
        self.ties = 0
        self.total_games = 0        
        
        # Pick who goes first
        if (random.random() < 0.5):
            self.player_turn = self.player_one
            self.turn = 'X'
        else:
            self.player_turn = self.player_two
            self.turn = 'O'
        
    def play_game(self):
        # play a game here
        self.show_board()         
        while self.winner == None:

            # TODO Human interaction goes here            

            # make a move
            self.state = self.make_move()
            self.check_for_winner()
            self.show_board()            
            # see if a winner exists
            if self.winner is not None:
                break
    
    
    def check_for_winner(self):
        
        # for each possible winning combination
        for combo in self.winning_combos:
            # Get the values from the board
            print(combo)
            print(self.state)
            if self.state:
                values = self.state[combo[0]] + self.state[combo[1]] + self.state[combo[2]]
            
                # X win
                if values == 'XXX': 
                    self.winner = 'X'
                    self.x_wins += 1
                    self.total_games += 1
                    print("X Wins")
                    break
                # O win
                elif values == 'OOO':  
                    self.winner = 'O'
                    self.o_wins += 1
                    self.total_games += 1
                    print("O Wins")
                    break
                # Tie (board is full)
                elif not any(val.isnumeric() for val in list(self.state)):
                    self.winner = 'No winner'
                    self.ties += 1
                    self.total_games += 1
                    print("Tie game")
        return self.winner
            
    
    def train(self, target, epochs):
        """Train the agents against each other
        """
        x_training = self.state_as_nums(self.prev_state)
        if target:

            # clear the board
            self.board = self.create_board()
            
            # play a game
            self.play_game()

        print("Agents trained.")

    def make_move(self, learning=False):
        """Agent makes a move
        """
        if self.turn == 'X':
            if learning == True:
                new_state = self.player_one.move_and_learn(self.state, self.winner)
            else:
                new_state = self.player_one.move(self.state, self.winner)
            
            # change turns
            self.turn = 'O'
            self.player_turn = self.player_two
        
        else:
            if learning == True:
                new_state = self.player_two.move_and_learn(self.state, self.winner)
            else:
                new_state = self.player_two.move(self.state, self.winner)
        
            # change turns
            self.turn = 'X'
            self.player_turn = self.player_one
        return new_state



    def show_board(self):
        """Displays the board
        """
        if self.state:
            top_row = "|".join([str(cell) for cell in self.state[0:3]])
            mid_row = "|".join([str(cell) for cell in self.state[3:6]])
            bot_row = "|".join([str(cell) for cell in self.state[6:9]])
            border = "-----"
       
            print('') 
            print(top_row)
            print(border)
            print(mid_row)
            print(border)
            print(bot_row)
            print('')
