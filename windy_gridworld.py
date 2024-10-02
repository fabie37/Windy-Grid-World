
import numpy as np


def init_windy_states():
    windy_states = np.zeros((7, 10), dtype=int)

    for col in range(3, 9):
        windy_states[:, col] += 1
    for col in range(6, 8):
        windy_states[:, col] += 1

    return windy_states


def stochastic_wind(wind):

    if not wind:
        return wind

    random_e = np.random.random()

    if random_e >= 2/3:
        return wind + 1
    if random_e <= 1/3:
        return max(wind - 1, 0)
    return wind


def policy(state, Q):
    random_e = np.random.random()

    if random_e > epsilon:
        return optimal(state, Q)
    return random(state, Q)


def optimal(state, Q):
    optimal_actions = np.where(Q[state] == np.max(Q[state]))[0]
    return int(np.random.choice(optimal_actions))


def random(state, Q):
    all_actions = [x for x in range(0, Q.shape[2])]
    return int(np.random.choice(all_actions))


def choose_start_state():
    return (3, 0)


def terminal(state):
    return state == (3, 7)


def move(state, action, stochastic=True):
    row, col = state
    wind = int(windy_states[row, col])
    if stochastic:
        wind = stochastic_wind(wind)
    steps_row, steps_col = steps(action)
    new_row = min(max(row+steps_row-wind, 0), windy_states.shape[0]-1)
    new_col = min(max(col+steps_col, 0), windy_states.shape[1]-1)
    new_state = (new_row, new_col)

    if terminal(new_state):
        return new_state, 0
    return new_state, -1


def steps(action):
    action_step_map = {
        0: (0, -1),
        1: (-1, -1),
        2: (-1, 0),
        3: (-1, 1),
        4: (0, 1),
        5: (1, 1),
        6: (1, 0),
        7: (1, -1)
    }
    return action_step_map[action]


windy_states = init_windy_states()
Q = np.zeros((7, 10, 8))
epsilon = 0.01
number_episodes = 10000
episodes_left = number_episodes
alpha = 0.5
discount = 1
avg_episode_steps = 0

while episodes_left > 0:

    state = choose_start_state()
    action = policy(state, Q)
    episode_steps = 0
    while not terminal(state):
        next_state, reward = move(state, action, stochastic=True)
        next_action = policy(next_state, Q)
        Q[state + (action,)] = Q[state + (action,)] + alpha * (reward +
                                                               (discount * Q[next_state + (next_action,)]) - Q[state + (action,)])
        state = next_state
        action = next_action
        episode_steps += 1
    episodes_left -= 1
    avg_episode_steps = avg_episode_steps + \
        (1/(number_episodes-episodes_left))*(episode_steps-avg_episode_steps)
    print(f"Average Steps per episode is: {avg_episode_steps}")
