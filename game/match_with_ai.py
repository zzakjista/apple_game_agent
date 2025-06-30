import pygame
import sys
import torch
from environment.environment import Environment
from agent.agent import Agent, state_from_env, get_valid_actions
from common.ui_utils import draw_env, get_cell_from_pos
from common.config import (
    GAME_MATRIX_COLS, 
    GAME_MATRIX_ROWS,
    MAX_ACTION_LEN,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    MODEL_PATH
)
# --- 초기화 ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Apple Game: Human vs AI')
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

# --- 환경 및 에이전트 ---
env = Environment(rows=GAME_MATRIX_ROWS, cols=GAME_MATRIX_COLS)
ai_agent = Agent(state_dim=GAME_MATRIX_ROWS*GAME_MATRIX_COLS*2, action_dim=MAX_ACTION_LEN*GAME_MATRIX_ROWS*GAME_MATRIX_COLS, rows=GAME_MATRIX_ROWS, cols=GAME_MATRIX_COLS, max_action_len=MAX_ACTION_LEN)
ai_agent.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
ai_agent.epsilon = 0.0  # 탐험 없이 greedy

# --- 상태 ---
current_player = 1  # 1: 사람, 2: AI
selected_cells = []
dragging = False
game_over = False

def reset_game():
    global env, current_player, selected_cells, dragging, game_over
    env = Environment(rows=GAME_MATRIX_ROWS, cols=GAME_MATRIX_COLS)
    current_player = 1
    selected_cells = []
    dragging = False
    game_over = False

def ai_turn():
    global current_player, game_over
    state = state_from_env(env)
    valid_actions = get_valid_actions(env)
    if not valid_actions:
        game_over = True
        return
    action = ai_agent.act(state, valid_actions)
    env.select_cells(action, player=2)
    current_player = 1
    if not get_valid_actions(env):
        game_over = True

def handle_human_action():
    global current_player, selected_cells, game_over
    if env.select_cells(selected_cells, player=1):
        current_player = 2
        selected_cells.clear()
        if not get_valid_actions(env):
            game_over = True
    else:
        selected_cells.clear()  # 잘못된 선택시 선택 해제

def skip_human_turn():
    global current_player, selected_cells
    # Simply pass the turn to AI, clear selection
    current_player = 2
    selected_cells.clear()

def match_with_ai():
    global dragging, selected_cells, current_player, game_over
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    reset_game()
                if not game_over and current_player == 1 and event.key == pygame.K_RETURN and selected_cells:
                    handle_human_action()
                # --- 스킵 기능: S 키로 턴 넘기기 ---
                if not game_over and current_player == 1 and event.key == pygame.K_s:
                    current_player = 2
                    selected_cells.clear()
            if not game_over and current_player == 1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    cell = get_cell_from_pos(event.pos, GAME_MATRIX_ROWS, GAME_MATRIX_COLS)
                    if cell:
                        dragging = True
                        selected_cells = [cell]
                if event.type == pygame.MOUSEBUTTONUP:
                    dragging = False
                if event.type == pygame.MOUSEMOTION and dragging:
                    cell = get_cell_from_pos(event.pos, GAME_MATRIX_ROWS, GAME_MATRIX_COLS)
                    if cell and cell not in selected_cells:
                        selected_cells.append(cell)
        if not game_over and current_player == 2:
            pygame.time.delay(400)
            ai_turn()
        draw_env(screen, font, env, selected_cells, game_over=game_over, current_player=current_player)
        clock.tick(30)
