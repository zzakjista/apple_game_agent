import os
import pygame
import torch
from environment.environment import Environment
from agent.agent import Agent, state_from_env, get_valid_actions
from common.ui_utils import draw_env
from common.config import (
    GAME_MATRIX_COLS,
    GAME_MATRIX_ROWS,
    MAX_ACTION_LEN
)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw_score(screen, font, episode, episodes, step, score1, score2):
    score_text = font.render(f"Ep {episode+1}/{episodes}  \n Step {step}  P1: {score1}  P2: {score2}", True, (0,0,0))
    screen.blit(score_text, (20, 20))

def load_agent_weights(agent, path, device):
    if os.path.exists(path):
        agent.model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded weights from {path}")

def get_scores(env):
    p1 = sum(1 for v in env.selected_cells.values() if v == 1)
    p2 = sum(1 for v in env.selected_cells.values() if v == 2)
    return p1, p2

def init_agents(env, save_path1, save_path2, device):
    agent1 = Agent(state_dim=env.rows * env.cols * 2,
                   action_dim=MAX_ACTION_LEN * env.rows * env.cols,
                   rows=env.rows, cols=env.cols, max_action_len=MAX_ACTION_LEN, device=device)
    agent2 = Agent(state_dim=env.rows * env.cols * 2,
                   action_dim=MAX_ACTION_LEN * env.rows * env.cols,
                   rows=env.rows, cols=env.cols, max_action_len=MAX_ACTION_LEN, device=device)
    agent1.model.to(device)
    agent2.model.to(device)
    agent1.target_model.to(device)
    agent2.target_model.to(device)
    load_agent_weights(agent1, save_path1, device)
    load_agent_weights(agent2, save_path2, device)
    return agent1, agent2

def run_step(env, agent, current_player, state, device):
    valid_actions = get_valid_actions(env)
    if not valid_actions:
        return None, None, (None, None), True  # No valid actions, end the game
    # state를 device로 이동
    state = torch.FloatTensor(state).to(device)
    action = agent.act(state, valid_actions)
    prev_score1, prev_score2 = get_scores(env)
    env.select_cells(action, player=current_player)
    next_state = state_from_env(env)
    new_score1, new_score2 = get_scores(env)
    reward1 = new_score1 - prev_score1
    reward2 = new_score2 - prev_score2
    return action, next_state, (reward1, reward2), False

def run_episode(env, agent1, agent2, screen, font, clock, episode, episodes, max_steps, device):
    env.reset()
    state = state_from_env(env)
    total_reward1, total_reward2 = 0, 0
    done, step, current_player = False, 0, 1
    while not done and step < max_steps:
        # 이벤트 처리 (GUI 모드에서만)
        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return None, None
        agent = agent1 if current_player == 1 else agent2
        step_result = run_step(env, agent, current_player, state, device)
        if step_result is None:
            break
        action, next_state, (reward1, reward2), done = step_result
        if done:
            break
        if current_player == 1:
            agent1.remember(state, action, reward1, next_state, done)
            total_reward1 += reward1
        else:
            agent2.remember(state, action, reward2, next_state, done)
            total_reward2 += reward2
        state = next_state
        agent1.model.to(device)
        agent2.model.to(device)
        agent1.target_model.to(device)
        agent2.target_model.to(device)
        agent1.replay(batch_size=64)
        agent2.replay(batch_size=64)
        # 화면 업데이트 (GUI 모드에서만)
        if screen is not None and font is not None and clock is not None:
            draw_env(screen, font, env)
            new_score1, new_score2 = get_scores(env)
            draw_score(screen, font, episode, episodes, step, new_score1, new_score2)
            pygame.display.flip()
            clock.tick(30)
        step += 1
        current_player = 2 if current_player == 1 else 1
    return total_reward1, total_reward2

def train_with_gui(episodes=100, max_steps=200, save_path1='result/agent1_final.pt', save_path2='result/agent2_final.pt'):
    device = get_device()
    print(f'디바이스:{device}')
    env = Environment(rows=GAME_MATRIX_ROWS, cols=GAME_MATRIX_COLS)
    agent1, agent2 = init_agents(env, save_path1, save_path2, device)
    pygame.init()
    screen = pygame.display.set_mode((800, 900))
    pygame.display.set_caption('Apple Game RL Training (2 Agents)')
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()
    for episode in range(episodes):
        result = run_episode(env, agent1, agent2, screen, font, clock, episode, episodes, max_steps, device)
        if result is None:
            return  # 조기 종료
        total_reward1, total_reward2 = result
        print(f"Episode {episode+1} | P1 reward: {total_reward1} | P2 reward: {total_reward2}")
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    torch.save(agent1.model.state_dict(), save_path1)
    torch.save(agent2.model.state_dict(), save_path2)
    print(f"Saved agent1 weights to {save_path1}")
    print(f"Saved agent2 weights to {save_path2}")
    pygame.quit()

def train(episodes=100, max_steps=200, save_path1='result/agent1_final.pt', save_path2='result/agent2_final.pt'):
    device = get_device()
    print(f'Using device: {device}')
    env = Environment(rows=GAME_MATRIX_ROWS, cols=GAME_MATRIX_COLS)
    agent1, agent2 = init_agents(env, save_path1, save_path2, device)
    for episode in range(episodes):
        total_reward1, total_reward2 = run_episode(env, agent1, agent2, None, None, None, episode, episodes, max_steps, device)
        if total_reward1 is None or total_reward2 is None:
            break
        print(f"Episode {episode+1} | P1 reward: {total_reward1} | P2 reward: {total_reward2}")
    torch.save(agent1.model.state_dict(), save_path1)
    torch.save(agent2.model.state_dict(), save_path2)
    print(f"Saved agent1 weights to {save_path1}")
    print(f"Saved agent2 weights to {save_path2}")

if __name__ == "__main__":
    train_with_gui(episodes=100, max_steps=200, save_path1='result/agent1_final.pt', save_path2='result/agent2_final.pt')

