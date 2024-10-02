# Windy Grid World 

This is a single solution to Sutton and Barto's Reinforcement Learning Exercises for on-policy TD learning (SARSA).

Problem: 
- Minimise steps required to reach terminal state in a grid world with windy columns
- A windy column has a value that moves the agent up a grid square by a certain value
- For Example: if state (4,5) is on a windy column of 2, then an agent on this state will move to (2,5)

RL Algorithm:
- Policy: fixed e-greedy
- SARSA: On-policy Tabluar TD

Requirements:
- pygame
- numpy

windy_gridworld.py:
- Base Implementation
- Gives the output of avg steps per episode

visualiser.py:
- Gives you a pygame visualiser of the algorithm
- It'll run the algorithm, learn the optimal policy then a continue button will be presented, once clicking, the greedy policy will be simulated