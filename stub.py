import numpy.random as npr
from matplotlib import pyplot as plt
import numpy as np
import sys

from SwingyMonkey import SwingyMonkey

def check_state(state, reduced_state):
    bot_tree = 0
    vel_down = 0
    tree_dist = 0
    monkey_pos_bot = 0

    bot_tree_pos = state['tree']['bot']
    if (bot_tree_pos < 25):
        bot_tree = 0
    elif (bot_tree_pos >= 25 and bot_tree_pos < 75):
        bot_tree = 1
    elif (bot_tree_pos >= 75 and bot_tree_pos < 125):
        bot_tree = 2
    else:
        bot_tree = 3


    vel_state = state['monkey']['vel']
    if (vel_state > 0):
        vel_down = 1

    # Checking the positioning of the monkey
    m_bot = state['monkey']['bot']
    if (m_bot < 75):
        monkey_pos_bot = 0
    elif (m_bot >= 75 and m_bot < 150):
        monkey_pos_bot = 1
    elif (m_bot >= 150 and m_bot < 225):
        monkey_pos_bot = 2
    else:
        monkey_pos_bot = 3

    # Checking the distance of money to tree
    dist = state['tree']['dist']
    if (dist < 100):
        tree_dist = 0
    elif (dist >= 100 and dist < 200):
        tree_dist = 1
    else:
        tree_dist = 2



    # Calculating the q state
    q_state = (4*vel_down) + bot_tree
    q_state += (32*tree_dist) + (8*monkey_pos_bot)

    if (reduced_state == False):
        top_tree_pos = state['tree']['top']
        if (top_tree_pos < 225):
            top_tree = 0
        elif (top_tree_pos >= 225 and top_tree_pos < 275):
            top_tree = 1
        elif (top_tree_pos >= 275 and top_tree_pos < 325):
            top_tree = 2
        else:
            top_tree = 3

        m_top = state['monkey']['top']
        if (m_top < 132):
            monkey_pos_top = 0
        elif (m_top >= 132 and m_top < 207):
            monkey_pos_top = 1
        elif (m_top >= 207 and m_top < 282):
            monkey_pos_top = 2
        else:
            monkey_pos_top = 3

        q_state += 384*monkey_pos_top + 96 * top_tree

    return q_state

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.q_matrix_reduced = np.zeros((96,2))
        self.q_matrix = np.zeros((1536, 2))
        self.alpha = .3
        self.gamma = 1#.53
        self.eplison = 0
        self.learning = True
        self.reduced = True

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.
        if (learner.reduced == True):
            q_matrix = self.q_matrix_reduced
        else:
            q_matrix = self.q_matrix
        alpha = self.alpha
        eplison = self.eplison
        gamma = self.gamma

        new_q_state = check_state(state, self.reduced)
        if (self.learning == True):
            if (self.last_state != None):
                # Finds the states in term of the q_matrix
                last_q_state = check_state(self.last_state, self.reduced)

                # Calculating old Q values
                old_q_value = (1-alpha) * q_matrix[last_q_state, self.last_action]

                # Calculating New Q-value
                q_array = q_matrix[new_q_state, :]
                q_best_action = np.argmax(q_array)
                best_q_value = q_matrix[new_q_state, q_best_action]
                new_q_value = alpha * (self.last_reward + gamma*best_q_value)

                # Updating q_values
                if (self.reduced == True):
                    self.q_matrix_reduced[last_q_state, self.last_action] = old_q_value + new_q_value
                else:
                    self.q_matrix[last_q_state, self.last_action] = old_q_value + new_q_value

                # Implement eplison greedy
                if (npr.rand() > eplison):
                    new_action = npr.rand() < 0.5
                else:
                    a = np.argmax(q_matrix[new_q_state, :])
                    new_action = a

            else:
                new_action = npr.rand() < .5
        else:
            new_action = np.argmax(q_matrix[new_q_state, :])

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        new_state  = state
        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

iters = 1000
max_score = 0
tot_score = 0
learner = Learner()

for ii in range(iters):
    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)
    # Loop until you hit something.
    while swing.game_loop():
        pass

    tot_score += swing.score
    if (swing.score > max_score):
        max_score = swing.score
    # Reset the state of the learner.
    learner.reset()
    learner.eplison = 1 - 1/(ii+1)


learner.learning = False
score = []

for i in range (100):
    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (i), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)
    # Loop until you hit something.
    while swing.game_loop():
        pass
    score.append(swing.score)
    learner.reset()

bins = list(range(0, 70))
print(learner.q_matrix_reduced)
print("Max Score")
print(max_score)
print("Average Score")
print(tot_score/iters)
plt.hist(score, bins)
plt.show()
print("Final Average Score: ")
print(sum(score)/len(score))
print("Max Score: ")
print(max(score))
