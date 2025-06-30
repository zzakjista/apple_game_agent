import pygame
from common.config import (
    MAX_ACTION_LEN,
    CELL_SIZE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    GRID_OFFSET_X,
    GRID_OFFSET_Y
)
from common.config import (
    WHITE,
    BLACK,
    GRAY,
    RED,
    BLUE,
    YELLOW
)


def draw_env(screen, font, env, selected_cells=None, info_text=None, game_over=False, current_player=1):
    screen.fill(GRAY)
    selected_cells = selected_cells or []
    for row in range(env.rows):
        for col in range(env.cols):
            x = GRID_OFFSET_X + col * CELL_SIZE
            y = GRID_OFFSET_Y + row * CELL_SIZE
            cell_owner = env.selected_cells.get((row, col))
            if (row, col) in selected_cells:
                color = YELLOW
            elif cell_owner == 1:
                color = BLUE
            elif cell_owner == 2:
                color = RED
            else:
                color = WHITE
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)
            if (row, col) not in env.selected_cells:
                number = font.render(str(env.matrix[row][col]), True, BLACK)
                rect = number.get_rect(center=(x + CELL_SIZE//2, y + CELL_SIZE//2))
                screen.blit(number, rect)
    # 점수/턴 표시
    p1 = sum(1 for v in env.selected_cells.values() if v == 1)
    p2 = sum(1 for v in env.selected_cells.values() if v == 2)
    info = f"P1: {p1}  P2: {p2}"
    if info_text:
        info += "  " + info_text
    info_render = font.render(info, True, BLACK)
    screen.blit(info_render, (20, 20))
    if game_over:
        if p1 > p2:
            result = "Player 1 Wins!"
        elif p2 > p1:
            result = "Player 2 Wins!"
        else:
            result = "Draw!"
        result_text = font.render(result, True, RED if p2 > p1 else BLUE if p1 > p2 else BLACK)
        screen.blit(result_text, (WINDOW_WIDTH//2 - result_text.get_width()//2, WINDOW_HEIGHT//2))
        info2 = font.render("Press R to restart, ESC to quit", True, BLACK)
        screen.blit(info2, (WINDOW_WIDTH//2 - info2.get_width()//2, WINDOW_HEIGHT//2 + 50))
    pygame.display.flip()

def get_cell_from_pos(pos, rows=10, cols=10):
    x, y = pos
    if (GRID_OFFSET_X <= x < GRID_OFFSET_X + cols * CELL_SIZE and
        GRID_OFFSET_Y <= y < GRID_OFFSET_Y + rows * CELL_SIZE):
        row = (y - GRID_OFFSET_Y) // CELL_SIZE
        col = (x - GRID_OFFSET_X) // CELL_SIZE
        return row, col
    return None