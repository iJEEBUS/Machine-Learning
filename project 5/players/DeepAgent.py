import random # to pick who goes first
import keras.layers as kl # for NN
import keras.models as km # for NN
import numpy as np
from . import Player  

class DeepAgent(Player.Player):
    """A Deep Learning Agent

    The bot version of a player that makes use of the Keras framework
    and reinforcement learning.

    Bots are trained by playing against each other and can play Humans.
    """
    def __init__(self, value, exploration_factor):
        super().__init__(value, exploration_factor)
        self.val = value
        self.val_model = self.create_model()
        self.epsilon = 0.1
        self.alpha = 0.5
        self.prev_state = '123456789'
        self.print = False
        
        if self.val == 'X':
            self.op_val = 'O'
        else:
            self.op_val = 'X'
    
    def create_model(self):
        """Create the NN
        """
        print('Creating new Neural Network')
        model = km.Sequential()
        model.add(kl.Dense(18, activation='relu', input_dim=9))
        model.add(kl.Dense(18, activation='relu'))
        model.add(kl.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
        model.summary()
        return model
    
    def move(self, state, winner):
        """Place a move on the boadd
        """
        self.state = state
        new_state = state 
        # Do not move if a winner is declared
        if winner is not None:
            new_state = state
            return state
        
        # Choose best move 90% of the time
        # Otherwise, randomly pick
        if random.random() < self.exploration_factor:
            new_state = self.best_move(state, winner)
        else:
            potential_moves = [cell for cell in state if cell.isnumeric()]
            i = random.choice(potential_moves)
            if state.find(i):
                i = int(i)
                new_state = state[:i] + self.val + state[i+1:] 
        return new_state

    def move_and_learn(self, state, winner):
        """Learn the state and move accordingly
        """
        self.learn_state(state, winner)
        return self.move(state, winner)

    def best_move(self, state, winner):
        """Play the best move
        """
        # TODO This method does not work correctly.
        # Does not replace the temp state, it simply appends to it
        # Need the state to be 9 chars long, this makes it 18.

        potential_moves = [(cell + 1) for cell, val in enumerate(state) if val.isnumeric()] 
        print(potential_moves) 
        # if only one move exists, take it
        if len(potential_moves) == 1:
            i = potential_moves[0]
            if (i in [1,2,3,4,5,6,7,8,9]):
                new_state = state[:i] + self.val + state[i + 1:] 
                return new_state

        all_states = []
        value = -999999.99999
        for i in potential_moves:
            
            temp_values = []
            i = int(i)
            print('MOVE TO MAKE: ', i) 
            # make a move 
            if i == 1:
                temp_state = self.val + state[i:]
            elif i == 9:
                temp_state = state[:i-1] + self.val
            else:
                temp_state = state[:i-1] + self.val + state[i:]

            print('temp_state: ', temp_state)
            print('val: ', self.val)
            
            # Calculate the value after the opponent tries each move
            opponent_moves = [cell for cell in temp_state if cell.isnumeric()]
            print('op_moves: ', opponent_moves)

            for op_move in opponent_moves:
                op_move = int(op_move)
                opponent_state = temp_state[:op_move-1] + self.op_val + temp_state[op_move:]          
                temp_values.append(self.calc_value(opponent_state))
            
            # Remove none values
            temp_values = [val for val in temp_values if val is not None]

            # best move has the smallest value 
            if len(temp_values) != 0:
                temp_values = np.min(temp_values)
            else:
                temp_values = 1

            # collect all possible best moves 
            if temp_values > value:
                all_states = [temp_state]
                value = temp_values
            else:
                all_states.append(temp_state)
       
            # pick one of the best moves
            if len(all_states) > 0:
                new_state = random.choice(all_states)
            else:
                print('There is no best move.')

    def reward(self, winner):
        """Reward the agent
        """
        if winner is self.val:
            reward = 1
        elif winner is 'No winner':
            reward = 0.5
        elif winner is None:
            reward = 0
        else:
            reward = -1
            
        return reward
    
    def state_as_nums(self, state):
        """Convert state
        Represent the state as all numbers (replace X and O with 1 and -1)
        """
        nums = []
        for i in state:
            if i == 'X':
                nums.append(1)
            elif i == 'O':
                nums.append(-1)
            else:
                nums.append(0)
        nums = np.array([nums])
        return nums
    
    def learn_state(self, state, winner):
        """Familiarize agent with this state
        """    
        # Calculate the goal
        target = self.calc_target(state, winner)
        
        # Train NN based on target
        self.train(target, 10)
        
        # Update state
        self.prev_state = state
        
    def calc_value(self, state):
        """Predict values of next moves
        """
        self.val_model.predict(self.state_as_nums(state))

    def calc_target(self, state, winner):
        """Calculate target values
        
        Calculates the target values that will be used to train the network
        """
        if self.tag in state:
            current_val = self.calc_value(self.prev_state)
            reward = self.reward(winner)

            if winner is None:
                next_state_value = self.calc_value(state)
            else:
                next_state_value = 0

            # Q-learning to set the targets
            target = np.array(current_val + self.alpha * (reward + next_state_value - current_val))
            return target


    def train(self, target, epochs):
        """Train the model
        """
        x = self.state_as_nums(self.prev_state)
        
        if target:
            self.model.fit(x, target, epochs=epochs, verbose=0)
        
    def save_model(self):
        """Write models out to file
        """
        f_name = 'model'+self.tag+'.nn'
        try:
            os.remove(f_name)
        except:
            pass
        self.val_model.save(f_name)
