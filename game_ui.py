import pygame
import sys
from environment.environment import Environment
from agent.agent import Agent, state_from_env, get_valid_actions
from common.config import (
    GAME_MATRIX_COLS,
    GAME_MATRIX_ROWS,
    MAX_ACTION_LEN
)

# Initialize Pygame
pygame.init()

# Constants (동적 계산)
CELL_SIZE = min(50, 800 // GAME_MATRIX_COLS, 800 // GAME_MATRIX_ROWS)  # 셀 크기 자동 조정
WINDOW_WIDTH = max(800, GAME_MATRIX_COLS * CELL_SIZE + 40)
WINDOW_HEIGHT = max(900, GAME_MATRIX_ROWS * CELL_SIZE + 200)
GRID_OFFSET_X = (WINDOW_WIDTH - (GAME_MATRIX_COLS * CELL_SIZE)) // 2
GRID_OFFSET_Y = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class GameUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Apple Game")
        self.env = Environment(rows=GAME_MATRIX_ROWS, cols=GAME_MATRIX_COLS)
        self.current_player = 1
        self.selected_cells = []
        self.scores = {1: 0, 2: 0}
        self.font = pygame.font.Font(None, 36)
        self.button_font = pygame.font.Font(None, 32)
        self.reset_button_rect = pygame.Rect(WINDOW_WIDTH - 140, 20, 120, 40)
        self.dragging = False
        self.drag_start = None
        self.drag_cells = set()

    def reset_game(self):
        """게임 상태를 초기화합니다."""
        self.env.reset()
        self.scores = {1: 0, 2: 0}
        self.current_player = 1
        self.selected_cells.clear()
        self.dragging = False
        self.drag_start = None
        self.drag_cells.clear()

    def get_cell_from_pos(self, pos):
        x, y = pos
        if (GRID_OFFSET_X <= x < GRID_OFFSET_X + self.env.cols * CELL_SIZE and
            GRID_OFFSET_Y <= y < GRID_OFFSET_Y + self.env.rows * CELL_SIZE):
            row = (y - GRID_OFFSET_Y) // CELL_SIZE
            col = (x - GRID_OFFSET_X) // CELL_SIZE
            return row, col
        return None

    def handle_mouse_down(self, pos):
        if self.reset_button_rect.collidepoint(pos):
            self.reset_game()
            return
        cell = self.get_cell_from_pos(pos)
        if cell:
            self.dragging = True
            self.drag_start = cell
            self.drag_cells = {cell}
            self.selected_cells = [cell]

    def handle_mouse_up(self, pos):
        self.dragging = False
        self.drag_start = None
        self.drag_cells.clear()

    def handle_mouse_motion(self, pos):
        if self.dragging:
            cell = self.get_cell_from_pos(pos)
            if cell and cell not in self.selected_cells:
                self.selected_cells.append(cell)
                self.drag_cells.add(cell)

    def handle_key(self, key):
        if key == pygame.K_RETURN and self.selected_cells:
            if self.env.select_cells(self.selected_cells, self.current_player):
                self.current_player = 3 - self.current_player
                self.selected_cells.clear()
        elif key == pygame.K_ESCAPE:
            self.selected_cells.clear()
        elif key == pygame.K_r:
            self.reset_game()

    def draw_cell(self, row: int, col: int):
        x = GRID_OFFSET_X + col * CELL_SIZE
        y = GRID_OFFSET_Y + row * CELL_SIZE
        cell_owner = self.env.selected_cells.get((row, col))
        if (row, col) in self.selected_cells:
            cell_color = YELLOW
        elif cell_owner == 1:
            cell_color = BLUE
        elif cell_owner == 2:
            cell_color = RED
        else:
            cell_color = WHITE
        pygame.draw.rect(self.screen, cell_color, (x, y, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)
        if (row, col) not in self.env.selected_cells:
            number = self.font.render(str(self.env.matrix[row][col]), True, BLACK)
            number_rect = number.get_rect(center=(x + CELL_SIZE//2, y + CELL_SIZE//2))
            self.screen.blit(number, number_rect)

    def draw_grid(self):
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                self.draw_cell(row, col)

    def draw_reset_button(self):
        pygame.draw.rect(self.screen, GRAY, self.reset_button_rect)
        pygame.draw.rect(self.screen, BLACK, self.reset_button_rect, 2)
        btn_text = self.button_font.render("RESET", True, BLACK)
        btn_rect = btn_text.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(btn_text, btn_rect)

    def draw_scores(self):
        self.scores[1] = sum(1 for v in self.env.selected_cells.values() if v == 1)
        self.scores[2] = sum(1 for v in self.env.selected_cells.values() if v == 2)
        score1 = self.font.render(f"Player 1: {self.scores[1]}", True, BLUE)
        score2 = self.font.render(f"Player 2: {self.scores[2]}", True, RED)
        current = self.font.render(f"Current Player: {self.current_player}", True, BLACK)
        self.screen.blit(score1, (20, 20))
        self.screen.blit(score2, (20, 50))
        self.screen.blit(current, (WINDOW_WIDTH//2 - 100, 20))
        self.draw_reset_button()

    def draw(self):
        self.screen.fill(GRAY)
        self.draw_grid()
        self.draw_scores()
        pygame.display.flip()

    def run(self):
        """메인 게임 루프"""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
            self.draw()

    # def run_ai_vs_ai(self, delay=500):
    #     agents = [Agent(state_dim=self.env.rows * self.env.cols * 2,
    #                     action_dim=5 * self.env.rows * self.env.cols,  # max_action_len * rows * cols
    #                     rows=self.env.rows, cols=self.env.cols, max_action_len=5),
    #               Agent(state_dim=self.env.rows * self.env.cols * 2,
    #                     action_dim=5 * self.env.rows * self.env.cols,
    #                     rows=self.env.rows, cols=self.env.cols, max_action_len=5)]
    #     turn = 0
    #     while True:
    #         player = (turn % 2) + 1
    #         agent = agents[turn % 2]
    #         state = state_from_env(self.env)
    #         valid_actions = get_valid_actions(self.env)
    #         if not valid_actions:
    #             break
    #         action = agent.act(state, valid_actions)
    #         self.env.select_cells(action, player)
    #         self.current_player = 3 - self.current_player
    #         self.selected_cells = []
    #         self.draw()
    #         pygame.time.delay(delay)
    #         turn += 1
    #     self.draw()  # 마지막 상태 표시
    #     # 최종 결과 화면 표시
    #     p1 = sum(1 for v in self.env.selected_cells.values() if v == 1)
    #     p2 = sum(1 for v in self.env.selected_cells.values() if v == 2)
    #     result = "Draw!"
    #     if p1 > p2:
    #         result = "Player 1 Wins!"
    #     elif p2 > p1:
    #         result = "Player 2 Wins!"
    #     score_text = self.font.render(f"Final Score: P1={p1}  P2={p2}", True, BLACK)
    #     result_text = self.font.render(result, True, RED if p2 > p1 else BLUE if p1 > p2 else BLACK)
    #     self.screen.blit(score_text, (WINDOW_WIDTH//2 - score_text.get_width()//2, WINDOW_HEIGHT//2 - 60))
    #     self.screen.blit(result_text, (WINDOW_WIDTH//2 - result_text.get_width()//2, WINDOW_HEIGHT//2))
    #     info_text = self.font.render("Press ESC to exit", True, BLACK)
    #     self.screen.blit(info_text, (WINDOW_WIDTH//2 - info_text.get_width()//2, WINDOW_HEIGHT//2 + 60))
    #     pygame.display.flip()
    #     waiting = True
    #     while waiting:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 pygame.quit()
    #                 sys.exit()
    #             if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
    #                 waiting = False