import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# =====================
# Action Encoding Utils
# =====================
def encode_action(action, rows=17, cols=10, max_action_len=5):
    """
    action: list of (row, col) tuples
    Returns a flattened one-hot vector of shape (max_action_len * rows * cols,)
    """
    vec = np.zeros((max_action_len, rows, cols), dtype=np.float32)
    for i, (r, c) in enumerate(action):
        if i >= max_action_len:
            break
        vec[i, r, c] = 1.0
    return vec.flatten()

# =============
# Q(s, a) Model
# =============
class DQN_Qsa(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # state: (batch, state_dim), action: (batch, action_dim)
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # (batch,)

# =============
# DQN Agent
# =============
class Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 rows=17, cols=10, max_action_len=5, device='cpu'):
        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim  # action encoding dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rows = rows
        self.cols = cols
        self.max_action_len = max_action_len
        # Experience replay
        self.memory = deque(maxlen=10000)
        # Networks
        self.model = DQN_Qsa(state_dim, action_dim).to(device)
        self.target_model = DQN_Qsa(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        """
        state: flattened state vector or tensor
        valid_actions: list of action (list of (r, c))
        Returns: action (list of (r, c))
        """
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        device = next(self.model.parameters()).device
        # state가 이미 tensor면 그대로, 아니면 변환
        if isinstance(state, torch.Tensor):
            state_tensor = state.to(device)
        else:
            state_tensor = torch.FloatTensor(state).to(device)
        state_tensor = state_tensor.unsqueeze(0) if state_tensor.dim() == 1 else state_tensor
        state_batch = state_tensor.repeat(len(valid_actions), 1)
        action_vecs = [encode_action(a, self.rows, self.cols, self.max_action_len) for a in valid_actions]
        action_batch = torch.FloatTensor(np.stack(action_vecs)).to(device)
        with torch.no_grad():
            q_values = self.model(state_batch, action_batch).cpu().numpy()
        best_idx = int(np.argmax(q_values))
        return valid_actions[best_idx]

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        device = next(self.model.parameters()).device
        minibatch = random.sample(self.memory, batch_size)
        # Prepare batches
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(encode_action(action, self.rows, self.cols, self.max_action_len))
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states = torch.FloatTensor(np.stack(states)).to(device)
        actions = torch.FloatTensor(np.stack(actions)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(device)
        dones = torch.BoolTensor(dones).to(device)
        # Compute targets
        target_qs = []
        for i in range(batch_size):
            if dones[i]:
                target_qs.append(rewards[i])
            else:
                valid_next_actions = self.get_valid_actions_for_state(next_states[i].detach().cpu().numpy())
                if not valid_next_actions:
                    target_qs.append(rewards[i])
                else:
                    next_action_vecs = [encode_action(a, self.rows, self.cols, self.max_action_len) for a in valid_next_actions]
                    ns = next_states[i].unsqueeze(0).repeat(len(next_action_vecs), 1).to(device)
                    na = torch.FloatTensor(np.stack(next_action_vecs)).to(device)
                    with torch.no_grad():
                        q_next = self.target_model(ns, na).max().item()
                    target_qs.append(rewards[i] + self.gamma * q_next)
        target_qs = torch.stack(target_qs) if isinstance(target_qs[0], torch.Tensor) else torch.FloatTensor(target_qs).to(device)
        pred_qs = self.model(states, actions)
        loss = nn.MSELoss()(pred_qs, target_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_valid_actions_for_state(self, state):
        """
        state: flattened state vector (2, rows, cols) -> (2*rows*cols,)
        환경을 복원하여 get_valid_actions(env)로 valid actions 반환
        """
        # state: [matrix_flat, owner_flat] concat
        mat_size = self.rows * self.cols
        mat = np.array(state[:mat_size]).reshape((self.rows, self.cols)).astype(int)
        owner = np.array(state[mat_size:]).reshape((self.rows, self.cols)).astype(int)
        # 환경 복원
        from environment.environment import Environment
        env = Environment(rows=self.rows, cols=self.cols)
        env.matrix = mat.tolist()
        env.selected_cells = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if owner[r, c] != 0:
                    env.selected_cells[(r, c)] = owner[r, c]
        return get_valid_actions(env, max_len=self.max_action_len)

# =====================
# State/Action Utilities
# =====================
def state_from_env(env):
    mat = np.array(env.get_matrix())
    owner = np.zeros_like(mat)
    for (r, c), p in env.selected_cells.items():
        owner[r, c] = p
    state = np.stack([mat, owner], axis=0).flatten()
    return state

def get_valid_actions(env, max_len=5):
    actions = set()
    matrix = env.get_matrix()
    rows, cols = env.rows, env.cols
    for r in range(rows):
        for c in range(cols):
            for h in range(1, max_len+1):
                for w in range(1, max_len+1):
                    # h*w > max_len 조건 제거: 직사각형(예: 3x2, 2x3)도 허용
                    if h * w > max_len:
                        continue  # 단, 셀 개수는 max_len 이하로 제한
                    if r + h > rows or c + w > cols:
                        continue
                    cells = [(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)]
                    if len(cells) > max_len:
                        continue  # 셀 개수 제한
                    total = sum(matrix[rr][cc] for rr, cc in cells)
                    if total != 10:
                        continue
                    cell_set = set(cells)
                    valid = True
                    for rr, cc in cells:
                        if matrix[rr][cc] == 0:
                            contact = 0
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = rr+dr, cc+dc
                                if (nr, nc) in cell_set and matrix[nr][nc] != 0:
                                    contact += 1
                            if contact < 2:
                                valid = False
                                break
                    if valid:
                        actions.add(tuple(sorted(cells)))
    return [list(action) for action in actions]
