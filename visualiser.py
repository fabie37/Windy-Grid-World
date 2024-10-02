import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 80
GRID_WIDTH = 10
GRID_HEIGHT = 7
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT + 50  # Extra space for button

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (200, 200, 200)

# Create the window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Stochastic Windworld RL Visualizer (SARSA)")

# Font for rendering text
font = pygame.font.Font(None, 30)


def init_windy_states():
    windy_states = np.zeros((7, 10), dtype=int)
    for col in range(3, 9):
        windy_states[:, col] += 1
    for col in range(6, 8):
        windy_states[:, col] += 1
    return windy_states


windy_states = init_windy_states()


def draw_grid():
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_HEIGHT - 50))
    for y in range(0, WINDOW_HEIGHT - 50, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WINDOW_WIDTH, y))


def draw_windy_squares():
    for col in range(GRID_WIDTH):
        for row in range(GRID_HEIGHT):
            wind_strength = windy_states[row, col]
            if wind_strength > 0:
                # Deeper blue for higher wind
                color = (0, 0, 255 - (wind_strength * 60))
                pygame.draw.rect(
                    screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            # Display wind strength
            text = font.render(str(wind_strength), True, BLACK)
            text_rect = text.get_rect(
                center=(col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2))
            screen.blit(text, text_rect)


def draw_goal():
    pygame.draw.rect(screen, GREEN, (7 * CELL_SIZE, 3 *
                     CELL_SIZE, CELL_SIZE, CELL_SIZE))


def draw_agent(state):
    pygame.draw.rect(screen, RED, (state[1] * CELL_SIZE + 5,
                     state[0] * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))


def draw_button(text):
    button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100,
                              WINDOW_HEIGHT - 40, 200, 30)
    pygame.draw.rect(screen, GRAY, button_rect)
    text_surf = font.render(text, True, BLACK)
    text_rect = text_surf.get_rect(center=button_rect.center)
    screen.blit(text_surf, text_rect)
    return button_rect


def choose_start_state():
    return (3, 0)


def terminal(state):
    return state == (3, 7)


def policy(state, Q, epsilon):
    if np.random.random() > epsilon:
        return int(np.argmax(Q[state]))
    return np.random.randint(8)


def greedy_policy(state, Q):
    return int(np.argmax(Q[state]))


def move(state, action):
    row, col = state
    wind = int(windy_states[row, col])
    if np.random.random() < 1/3:
        wind = max(wind - 1, 0)
    elif np.random.random() < 2/3:
        wind += 1

    action_step_map = {
        0: (0, -1), 1: (-1, -1), 2: (-1, 0), 3: (-1, 1),
        4: (0, 1), 5: (1, 1), 6: (1, 0), 7: (1, -1)
    }
    steps_row, steps_col = action_step_map[action]

    new_row = min(max(row + steps_row - wind, 0), GRID_HEIGHT - 1)
    new_col = min(max(col + steps_col, 0), GRID_WIDTH - 1)
    return (new_row, new_col)


def main():
    clock = pygame.time.Clock()
    Q = np.zeros((GRID_HEIGHT, GRID_WIDTH, 8))
    epsilon = 0.2
    alpha = 0.10
    gamma = 1.0
    episodes = 100000

    for episode in range(episodes):
        state = choose_start_state()
        action = policy(state, Q, epsilon)  # Choose initial action
        steps = 0

        while not terminal(state):
            next_state = move(state, action)
            # Choose next action using the policy
            next_action = policy(next_state, Q, epsilon)
            reward = -1 if not terminal(next_state) else 0

            # SARSA update
            Q[state + (action,)] += alpha * (reward + gamma *
                                             Q[next_state + (next_action,)] - Q[state + (action,)])

            state = next_state
            action = next_action  # Update action for the next iteration
            steps += 1

        print(f"Episode {episode + 1} completed in {steps} steps")

    # Learning phase complete, wait for user to continue
    waiting = True
    while waiting:
        screen.fill(WHITE)
        draw_windy_squares()
        draw_grid()
        draw_goal()
        button_rect = draw_button("Continue")
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    waiting = False

    # Demonstrate learned policy
    while True:
        state = choose_start_state()
        steps = 0

        while not terminal(state):
            screen.fill(WHITE)
            draw_windy_squares()
            draw_grid()
            draw_goal()
            draw_agent(state)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = greedy_policy(state, Q)  # Use greedy policy
            state = move(state, action)
            steps += 1

            pygame.display.flip()
            clock.tick(10)  # Slow down the demonstration

        print(f"Greedy policy completed in {steps} steps")


if __name__ == "__main__":
    main()
