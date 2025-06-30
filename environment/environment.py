import random
from typing import List, Tuple

class Environment:
    def __init__(self, rows: int = 10, cols: int = 17):
        self.rows = rows
        self.cols = cols
        self.matrix = self._generate_matrix()
        self.selected_cells = {}  # {(r, c): player}

    def _generate_matrix(self) -> List[List[int]]:
        return [[random.randint(1, 9) for _ in range(self.cols)] for _ in range(self.rows)]

    def get_matrix(self) -> List[List[int]]:
        return self.matrix

    def reset(self):
        self.matrix = self._generate_matrix()
        self.selected_cells.clear()

    def is_valid_selection(self, cells: List[Tuple[int, int]]) -> bool:
        if self.check_cells_empty(cells):
            return False
        if not self.check_cells_sum(cells, 10):
            return False
        if not self.check_cells_bounds(cells):
            return False
        if not self.check_cells_shape(cells):
            return False
        if not self.check_zero_contact(cells):
            return False
        return True
    
    def check_cells_empty(self, cells: List[Tuple[int, int]]) -> bool:
        if not cells:
            return True
        return False

    def check_cells_sum(self, cells: List[Tuple[int, int]], check_value: int) -> bool:
        total = sum(self.matrix[r][c] for r, c in cells)
        if total == check_value:
            return True
        return False
    
    def check_cells_bounds(self, cells: List[Tuple[int, int]]) -> bool:
        for r, c in cells:
            if (0 <= r < self.rows and 0 <= c < self.cols):
                return True
        return False
    
    def check_cells_shape(self, cells: List[Tuple[int, int]]) -> bool:
        """
        선택된 셀이 직선(가로/세로) 또는 2x2 이상 직사각형인지 판별합니다.
        ㄱ자(꺾인) 모양 등은 허용하지 않습니다.
        """
        if not cells:
            return False
        rows = sorted(set(r for r, c in cells))
        cols = sorted(set(c for r, c in cells))
        # 직선(가로 또는 세로)
        if len(rows) == 1:
            # 한 행에 여러 열: 가로 직선
            if len(cols) == len(cells):
                return True
            else:
                return False
        if len(cols) == 1:
            # 한 열에 여러 행: 세로 직선
            if len(rows) == len(cells):
                return True
            else:
                return False
        # 2x2 이상 직사각형
        min_r, max_r = rows[0], rows[-1]
        min_c, max_c = cols[0], cols[-1]
        rect_cells = set((r, c) for r in range(min_r, max_r+1) for c in range(min_c, max_c+1))
        if len(cells) == len(rect_cells) and (max_r-min_r+1 >= 2 and max_c-min_c+1 >= 2):
            if set(cells) == rect_cells:
                return True
        return False
    
    def check_zero_contact(self, cells: List[Tuple[int, int]]) -> bool:
        cell_set = set(cells)
        zero_contact = 0
        for r, c in cells:
            if self.matrix[r][c] == 0:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) in cell_set and self.matrix[nr][nc] != 0:
                        zero_contact += 1
        if any(self.matrix[r][c] == 0 for r, c in cells):
            if zero_contact < 2:
                return False
        return True

    def select_cells(self, cells: List[Tuple[int, int]], player: int):
        """
        선택된 셀을 해당 플레이어의 것으로 처리하고, 값을 0으로 만듭니다.
        유효성 검사는 is_valid_selection에서 이미 수행됨을 전제합니다.
        """
        if not self.is_valid_selection(cells):
            return False
        for r, c in cells:
            self.selected_cells[(r, c)] = player
            self.matrix[r][c] = 0
        return True